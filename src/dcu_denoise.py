"""Isolated original DCU NoiseCut module, using its unchanged module defaults.

This is one original RGB16 spatial module. It is not the entire DCU RAW/GUI
denoising pipeline and does not claim to reproduce its four controls. The native
EXE is version/hash checked by Host; the GUI entry point is never invoked.
"""
from pathlib import Path
import ctypes as C
import json,os,subprocess,sys,time
import numpy as np
from temp_storage import temporary_directory,ensure_free_space,subprocess_environment

NATIVE_DEFAULT_WORDS=(128,30,128,0,30)


def _cancel(cancelled):
    if cancelled and cancelled():
        from native_backend import RenderCancelled
        raise RenderCancelled()


def _temporary_directory(required):
    return temporary_directory(required,prefix='dcu-denoise-')


def _stop(child,work):
    # Only the process tree created for this request is terminated.
    if child.poll() is not None:return
    # Prefer an orderly stop at a native row boundary so native working memory
    # and mapped result files are released in the worker's finally block.
    (work/'cancel').touch()
    try:child.wait(timeout=2);return
    except subprocess.TimeoutExpired:pass
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,timeout=15,check=False)
    else:child.terminate()
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)


def apply(rgb,cancelled=None,progress=None):
    """Run the original default NoiseCut in a disposable, headless CPU process."""
    if rgb.dtype not in (np.uint8,np.uint16) or rgb.ndim!=3 or rgb.shape[2]!=3 or not all(rgb.shape[:2]):
        raise ValueError('DCU 降噪需要非空 RGB8 / RGB16')
    _cancel(cancelled)
    if progress:progress('DCU 降噪 0% · 准备完整尺寸输入')
    # Disk-backed exchange avoids copying large arrays into a command line or
    # a pipe; neither source filenames nor source photo metadata are retained.
    with _temporary_directory(rgb.nbytes*2+64*1024*1024) as folder:
        work=Path(folder);source=work/'input.npy';output=work/'output.npy'
        np.save(source,rgb,allow_pickle=False);_cancel(cancelled)
        request=work/'request.json'
        request.write_text(json.dumps({'input':'input.npy','output':'output.npy','shape':list(rgb.shape),'dtype':rgb.dtype.name}),encoding='utf-8')
        if getattr(sys,'frozen',False):command=[sys.executable,'--dcu-denoise-worker',str(request)]
        else:command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(request)]
        child=None;last=''
        try:
            ensure_free_space(work,rgb.nbytes+64*1024*1024)
            child=subprocess.Popen(command,cwd=work,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                env=subprocess_environment(work))
            started=time.monotonic()
            while child.poll() is None:
                _cancel(cancelled)
                if time.monotonic()-started>1800:raise RuntimeError('DCU 降噪处理超时，本次任务已停止')
                try:
                    stage=json.loads((work/'stage.json').read_text(encoding='utf-8'))
                    label=stage['text']
                    if progress and label!=last:progress(label);last=label
                except (OSError,ValueError,KeyError):pass
                time.sleep(.08)
            _cancel(cancelled)
            if child.returncode or not output.is_file():raise RuntimeError('DCU 降噪未完成，请检查完整程序后重试')
            info=json.loads((work/'result.json').read_text(encoding='utf-8'))
            if (info.get('native_defaults')!=list(NATIVE_DEFAULT_WORDS) or info.get('shape')!=list(rgb.shape)
                    or info.get('dtype')!=rgb.dtype.name or info.get('foreground_calls')):
                raise RuntimeError('DCU 降噪结果校验失败')
            result=np.load(output,allow_pickle=False)
            if result.shape!=rgb.shape or result.dtype!=rgb.dtype:raise RuntimeError('DCU 降噪输出格式错误')
            _cancel(cancelled)
            if progress:progress('DCU 降噪 100% · 完成')
            return result
        finally:
            if child is not None:_stop(child,work)


def _write_stage(work,percent,text):
    # The parent's short read can temporarily deny replacement on Windows.
    # Progress is advisory; a missed update must never abort native processing.
    # The parent only declares completion after validating output + result.json.
    try:
        tmp=work/'stage.tmp';tmp.write_text(json.dumps({'percent':percent,'text':f'DCU 降噪 {percent}% · {text}'}),encoding='utf-8')
        os.replace(tmp,work/'stage.json')
        return True
    except PermissionError:return False


def _worker(request_file):
    from native_host import Host
    request_file=Path(request_file).resolve();work=request_file.parent
    request=json.loads(request_file.read_text(encoding='utf-8'))
    if request.get('input')!='input.npy' or request.get('output')!='output.npy':raise ValueError('Invalid request')
    source=work/'input.npy';destination=work/'output.npy'
    if destination.exists():raise ValueError('Output already exists')
    os.environ['TEMP']=os.environ['TMP']=str(work)
    rgb=np.load(source,allow_pickle=False,mmap_mode='r')
    out=None;host=None;obj=None;configured=False;success=False
    started=time.monotonic()
    try:
        if (rgb.dtype not in (np.uint8,np.uint16) or rgb.ndim!=3 or rgb.shape[-1]!=3
                or list(rgb.shape)!=request['shape'] or rgb.dtype.name!=request['dtype'] or not all(rgb.shape[:2])):
            raise ValueError('Invalid RGB buffer')
        h,w=rgb.shape[:2]
        _write_stage(work,0,'加载原厂 CPU 模块')
        host=Host()
        if not host.call(0x13a2634,C.c_int,[C.c_int],1):raise RuntimeError('Native CRT setup failed')
        if not host.call(0x13a25b0,C.c_int,[]):raise RuntimeError('Native CRT setup failed')
        host.initialize_full_crt()
        status=host.call(0xae3fb0,C.c_int16,
            [C.c_byte,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,
             C.c_byte,C.c_byte,C.c_byte,C.c_byte,C.c_void_p,C.c_void_p],1,0,0,0,2,0,1,1,1,1,0,0)
        if status:raise RuntimeError('Native initialization failed')
        storage=C.create_string_buffer(0x32d8);obj=C.addressof(storage)
        host.call(0xb46b50,C.c_void_p,[C.c_void_p],obj)
        configured=True
        if host.call(0xb46ba0,C.c_int16,[C.c_void_p,C.c_void_p],obj,0):raise RuntimeError('Native module init failed')
        params=C.create_string_buffer(0x110)
        host.call(0xb46be0,None,[C.c_void_p,C.c_void_p],obj,C.addressof(params))
        defaults=list(C.cast(C.addressof(params),C.POINTER(C.c_int32))[:5])
        if defaults!=list(NATIVE_DEFAULT_WORDS):raise RuntimeError('Native defaults changed')
        image_info=C.create_string_buffer(300)
        for offset,value in ((8,w),(12,h),(24,0x40)):C.c_int.from_buffer(image_info,offset).value=value
        packed=np.zeros((w,4),np.uint16);cursor=[0];callback_errors=[]

        @C.WINFUNCTYPE(C.c_int16,C.c_void_p,C.c_void_p)
        def read_row(user,dest):
            try:
                # The native seven-row reader replicates top/bottom borders;
                # it preloads four rows even if a diagnostic image is shorter.
                row=min(cursor[0],h-1);packed[:,:3]=rgb[row,:,::-1]
                if rgb.dtype==np.uint8:packed[:,:3]*=257
                C.memmove(dest,int(packed.ctypes.data),w*8);cursor[0]+=1;return 0
            except Exception:
                callback_errors.append(True);return -1

        status=host.call(0xb46c30,C.c_int16,[C.c_void_p]*5,obj,C.addressof(image_info),0,
                         C.cast(read_row,C.c_void_p).value,C.addressof(params))
        if status:raise RuntimeError('Native setup failed')
        out=np.lib.format.open_memmap(destination,mode='w+',dtype=rgb.dtype,shape=rgb.shape)
        line=np.full((w+2,4),0xa55a,np.uint16)
        run=C.WINFUNCTYPE(C.c_int16,C.c_void_p,C.c_void_p,C.c_void_p)(host.base+0xb46dd0)
        last_report=0.0
        for row in range(h):
            if (work/'cancel').exists():raise RuntimeError('Cancelled')
            if run(obj,0,int(line[1:].ctypes.data)):raise RuntimeError('Native row processing failed')
            if np.any(line[0]!=0xa55a) or np.any(line[-1]!=0xa55a):raise RuntimeError('Native buffer bounds failed')
            pixels=line[1:-1,:3][:,::-1]
            out[row]=pixels if rgb.dtype==np.uint16 else ((pixels.astype(np.uint32)+128)//257).astype(np.uint8)
            now=time.monotonic()
            if now-last_report>.12:
                _write_stage(work,min(99,(row+1)*100//h),'原厂默认设置');last_report=now
        if callback_errors or cursor[0]!=max(h,4):raise RuntimeError('Native row input failed')
        if host.blocked_ui:raise RuntimeError('Unexpected native foreground call')
        out.flush();out._mmap.close();out=None
        (work/'result.json').write_text(json.dumps({'shape':list(rgb.shape),'dtype':rgb.dtype.name,
            'native_defaults':defaults,'native_module':'DCU NoiseCut','full_gui_equivalence':False,
            'foreground_calls':False,'seconds':round(time.monotonic()-started,4)}),encoding='utf-8')
        # Only the parent may publish 100%, after the child exits cleanly and
        # both the result description and the pixel buffer have been checked.
        _write_stage(work,99,'结果校验');success=True
    finally:
        pending_error=sys.exc_info()[0] is not None
        cleanup_errors=[]
        try:
            if configured and host is not None:
                # Original working-buffer destructor, not operator delete on
                # the ctypes-owned object storage.
                host.call(0xb47ab0,C.c_int16,[C.c_void_p],obj)
        except Exception as exc:cleanup_errors.append(exc)
        finally:
            try:
                if out is not None:out._mmap.close()
            except Exception as exc:cleanup_errors.append(exc)
            finally:
                try:rgb._mmap.close()
                except Exception as exc:cleanup_errors.append(exc)
                finally:
                    if not success:
                        try:destination.unlink(missing_ok=True)
                        except Exception as exc:cleanup_errors.append(exc)
        if cleanup_errors:
            if pending_error:print('Native cleanup also failed: '+type(cleanup_errors[0]).__name__,file=sys.stderr)
            else:raise RuntimeError('Native cleanup failed') from cleanup_errors[0]


def worker_main(request_file):
    """CLI entry: --dcu-denoise-worker REQUEST; returns an exit status."""
    work=Path(request_file).resolve().parent
    with (work/'worker.log').open('w',encoding='utf-8') as log:
        sys.stdout=sys.stderr=log
        try:_worker(request_file)
        except Exception:
            import traceback
            traceback.print_exc();return 1
    return 0


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--worker':raise SystemExit(2)
    raise SystemExit(worker_main(sys.argv[2]))
