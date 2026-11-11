"""Persistent headless AI process: native session initialization never owns Tk's GIL."""
from pathlib import Path
import atexit,json,os,queue,subprocess,sys,threading,time
from collections import deque
import numpy as np
from temp_storage import temporary_directory,subprocess_environment

_child=None;_messages=None;_storage=None;_lock=threading.Lock();_lifecycle=threading.RLock()


class WorkerExited(RuntimeError):
    def __init__(self,child,status):
        code=child.poll()
        if code is None:
            try:code=child.wait(timeout=.25)
            except subprocess.TimeoutExpired:pass
        self.exit_code=code
        self.status=dict(status)
        self.stderr=list(getattr(child,'_ai_stderr',()))
        value='未知' if code is None else f'0x{code & 0xffffffff:08X}'
        self.code_text=value
        device=status.get('backend','后台')
        super().__init__(f'AI {device}进程退出（{value}）')


def _record_exit(exc,model):
    from neural_devices import record_failure,write_diagnostic
    from app_paths import CONFIG
    info=exc.status
    if (exc.exit_code is not None and (exc.exit_code & 0x80000000)
        and info.get('provider')=='DmlExecutionProvider' and info.get('adapter')):
        record_failure(info['adapter'],info.get('model',model),info.get('variant','base'),
                       RuntimeError('DirectML worker native exit '+exc.code_text))
    write_diagnostic({'model':model,'exit_code':exc.code_text,'stage':info,
                      'stderr':exc.stderr},CONFIG/'ai-worker-failure.json')

def _restore_worker_stdio():
    """Recover inherited Popen pipes in a Windows windowed executable.

    A console=False runtime may leave Python's stdio objects as None. Each
    recovered stream owns a DuplicateHandle copy, never the original handle.
    """
    if all(getattr(sys,name) is not None for name in ('stdin','stdout','stderr')):return
    if os.name!='nt':raise RuntimeError('AI worker standard pipes are unavailable')
    import ctypes
    from ctypes import wintypes
    import io
    import msvcrt
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetStdHandle.argtypes=[wintypes.DWORD];kernel.GetStdHandle.restype=wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.DuplicateHandle.argtypes=[wintypes.HANDLE,wintypes.HANDLE,wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.DuplicateHandle.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    current=kernel.GetCurrentProcess();created={}
    try:
        for name,number,mode,flags in (('stdin',-10,'rb',os.O_RDONLY),('stdout',-11,'wb',os.O_WRONLY),('stderr',-12,'wb',os.O_WRONLY)):
            if getattr(sys,name) is not None:continue
            original=kernel.GetStdHandle(wintypes.DWORD(number))
            if not original or original==ctypes.c_void_p(-1).value:
                raise RuntimeError('AI worker '+name+' pipe handle is unavailable')
            duplicate=wintypes.HANDLE()
            if not kernel.DuplicateHandle(current,original,current,ctypes.byref(duplicate),0,False,2):
                raise ctypes.WinError(ctypes.get_last_error())
            try:descriptor=msvcrt.open_osfhandle(duplicate.value,flags|os.O_BINARY)
            except BaseException:
                kernel.CloseHandle(duplicate);raise
            try:raw=io.FileIO(descriptor,mode=mode,closefd=True)
            except BaseException:
                os.close(descriptor);raise
            try:created[name]=io.TextIOWrapper(raw,encoding='utf-8',newline='\n',line_buffering=True,write_through=True)
            except BaseException:
                raw.close();raise
    except BaseException:
        for stream in created.values():stream.close()
        raise
    for name,stream in created.items():setattr(sys,name,stream)


def _cancelled(callback):
    if callback and callback():
        from native_backend import RenderCancelled
        raise RenderCancelled()

def shutdown(expected_child=None):
    global _child,_storage,_messages
    # Detach the process and its own directory atomically. A concurrently
    # started replacement must never have its storage removed by this call.
    with _lifecycle:
        if expected_child is not None and _child is not expected_child:return
        child=_child;storage=_storage
        _child=None;_storage=None;_messages=None
    try:
        if child is not None:
            if child.poll() is None:
                try:child.terminate()
                except OSError:
                    if child.poll() is None:raise
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
            for stream in (child.stdin,child.stdout,getattr(child,'stderr',None)):
                try:stream.close()
                except (OSError,AttributeError):pass
    finally:
        if storage is not None:storage.cleanup()

def _start():
    global _child,_messages,_storage
    with _lifecycle:
        if _child is not None and _child.poll() is None:return _child,_messages
        shutdown()
        storage=temporary_directory(256*1024*1024,prefix='photo-ai-host-')
        # Registered after this storage's weakref finalizer, so Windows stops
        # the process before attempting to remove its working directory.
        atexit.register(shutdown)
        if getattr(sys,'frozen',False):command=[sys.executable,'--ai-denoise-worker']
        else:command=[sys.executable,str(Path(__file__).resolve()),'--worker']
        env=subprocess_environment(storage.name);env['PHOTO_AI_WORKER']='1';env['PYTHONIOENCODING']='utf-8'
        try:
            child=subprocess.Popen(command,cwd=storage.name,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',bufsize=1,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=env)
        except BaseException:
            storage.cleanup();raise
        messages=queue.Queue();_child=child;_messages=messages;_storage=storage
        child._ai_stderr=deque(maxlen=24)
        def read_errors():
            try:
                for line in child.stderr:child._ai_stderr.append(line.rstrip()[:2048])
            except (OSError,ValueError):pass
        threading.Thread(target=read_errors,daemon=True,name='AIDenoiseDiagnostics').start()
        def read():
            try:
                for line in child.stdout:
                    try:messages.put(json.loads(line))
                    except ValueError:continue
            except (OSError,ValueError):pass
            finally:messages.put({'type':'exit'})
        threading.Thread(target=read,daemon=True,name='AIDenoiseMessages').start()
        return child,messages

def _infer_once(rgb,model,mode,cancelled,progress,on_status,variant=None,skip_devices=()):
    _cancelled(cancelled)
    if progress:progress('准备模型预热…' if rgb is None else '正在准备 AI 图像数据…')
    with temporary_directory((0 if rgb is None else rgb.nbytes*2)+64*1024*1024,prefix='photo-ai-job-') as folder:
        work=Path(folder)
        if rgb is not None:np.save(work/'input.npy',rgb,allow_pickle=False)
        _cancelled(cancelled)
        if progress:progress('正在准备 AI 后台运行环境…')
        child,messages=_start()
        job=work.name;terminal=False;last_status={}
        try:
            try:
                child.stdin.write(json.dumps({'job':job,'work':str(work),'model':model,'mode':mode,'variant':variant,
                    'skip_devices':list(skip_devices),
                    'operation':'warmup' if rgb is None else 'infer'})+'\n');child.stdin.flush()
            except (BrokenPipeError,OSError):
                if child.poll() is not None:raise WorkerExited(child,last_status) from None
                raise
            stop_since=None;started=time.monotonic()
            while True:
                if cancelled and cancelled():
                    if stop_since is None:(work/'cancel').touch();stop_since=time.monotonic()
                    if time.monotonic()-stop_since>1.5:
                        shutdown(expected_child=child);_cancelled(cancelled)
                try:message=messages.get(timeout=.04)
                except queue.Empty:
                    if child.poll() is not None:
                        _cancelled(cancelled);raise WorkerExited(child,last_status)
                    if time.monotonic()-started>7200:raise RuntimeError('AI 后台任务超时')
                    continue
                kind=message.get('type')
                if kind=='exit':_cancelled(cancelled);raise WorkerExited(child,last_status)
                if message.get('job')!=job:continue
                if kind=='progress':
                    if isinstance(message.get('status'),dict):
                        last_status=message['status']
                        if on_status:on_status(last_status)
                    if progress and stop_since is None:progress(message.get('text',''))
                elif kind=='result':
                    terminal=True;_cancelled(cancelled)
                    if rgb is None:return None,message['status']
                    if progress:progress('正在读取降噪结果…')
                    result=np.load(work/'output.npy',allow_pickle=False)
                    _cancelled(cancelled)
                    if result.shape!=rgb.shape or result.dtype!=rgb.dtype:raise RuntimeError('AI 后台返回的图像格式不匹配')
                    return result,message['status']
                elif kind=='cancelled':terminal=True;_cancelled(lambda:True)
                elif kind=='error':terminal=True;raise RuntimeError(message.get('error','AI 后台处理失败'))
        except BaseException:
            # A native crash, cancellation or callback failure must release the
            # dead process and its mappings before a new request can be started.
            if not terminal:shutdown(expected_child=child)
            raise


def infer(rgb,model,mode,cancelled=None,progress=None,on_status=None):
    import startup_prewarm
    startup_prewarm.before_infer(model,mode)
    waiting=False
    while not _lock.acquire(timeout=.04):
        _cancelled(cancelled)
        if progress:
            warm=startup_prewarm.snapshot()
            if warm.get('state')=='running':progress(warm.get('stage','准备降噪模型…'))
            elif not waiting:progress('等待前一个 AI 任务退出…')
        waiting=True
    busy=False
    try:
        _cancelled(cancelled)
        import gpu_compute
        gpu_compute.set_external_busy(True);busy=True
        selected=mode
        variant=None;recovery=None
        from neural_devices import adapter_key,priority_rows
        # Each native GPU exit quarantines one device/profile before the next
        # worker starts. Leave room to reach every configured GPU and CPU.
        try:attempts=max(2,len(priority_rows())) if selected=='auto' else 1
        except (OSError,RuntimeError):attempts=2
        skip_devices=set()
        for attempt in range(attempts):
            try:
                result,info=_infer_once(rgb,model,selected,cancelled,progress,on_status,variant,
                                        skip_devices)
                if recovery:
                    info=dict(info,fallback_kind='worker_exit',fallback_reason=recovery)
                return result,info
            except WorkerExited as exc:
                _record_exit(exc,model)
                native_gpu=(exc.exit_code is not None and (exc.exit_code & 0x80000000)
                            and exc.status.get('provider')=='DmlExecutionProvider')
                if attempt+1>=attempts or selected=='cpu' or not native_gpu:raise
                _cancelled(cancelled)
                adapter=exc.status.get('adapter')
                if adapter:skip_devices.add(adapter.get('key') or adapter_key(adapter))
                selected='auto' if adapter else 'cpu'
                if exc.status.get('model')==model:variant=exc.status.get('variant','base')
                recovery=('GPU 后台异常（'+exc.code_text+'），已跳过故障设备并按优先级重试'
                          if adapter else 'GPU 后台异常（'+exc.code_text+'），设备信息不可用，改用 CPU 处理')
                fallback_backend='按优先级重试' if adapter else 'CPU'
                if on_status:on_status({'model':model,'backend':fallback_backend,'phase':'recovering_backend',
                    'fallback_kind':'worker_exit','fallback_reason':recovery,'eta_seconds':None})
                if progress:progress(recovery+'…')
    finally:
        try:
            if busy:gpu_compute.set_external_busy(False)
        finally:_lock.release()


def worker_main():
    _restore_worker_stdio()
    try:
        import faulthandler
        faulthandler.enable(file=sys.stderr)
    except (OSError,ValueError,AttributeError):pass
    os.environ['PHOTO_AI_WORKER']='1'
    import denoise
    from native_backend import RenderCancelled
    def send(value):print(json.dumps(value,ensure_ascii=True),flush=True)
    def snapshot():
        info=denoise.status()
        if info.get('phase')=='processing_tiles':info['tile_elapsed_seconds']=info.get('elapsed_seconds')
        info['elapsed_seconds']=round(time.monotonic()-job_started,1)
        info['elapsed_scope']='worker job, including model preparation'
        return info
    def progress(text):send({'type':'progress','job':job,'text':text,'status':snapshot()})
    for line in sys.stdin:
        source=None;clean=None;job=None;response=None;job_started=time.monotonic()
        try:
            request=json.loads(line);job=request.get('job');work=Path(request['work'])
            def cancelled():return (work/'cancel').exists()
            if request.get('operation')=='warmup':
                if os.name=='nt':
                    import ctypes as C
                    C.windll.kernel32.SetPriorityClass(C.windll.kernel32.GetCurrentProcess(),0x4000)
                denoise.session(request['model'],force_cpu=request.get('mode')=='cpu',
                                cancelled=cancelled,progress=progress,
                                skip_devices=request.get('skip_devices',()))
            else:
                source=np.load(work/'input.npy',mmap_mode='r',allow_pickle=False)
                clean=denoise.apply(source,1.,cancelled,progress,model=request['model'],
                                    variant=request.get('variant'),force_cpu=request.get('mode')=='cpu',
                                    skip_devices=request.get('skip_devices',()))
                _cancelled(cancelled)
                denoise.report_phase('writing_result',eta_seconds=None)
                progress('正在传回降噪结果…')
                np.save(work/'output.npy',clean,allow_pickle=False)
            # The parent owns the reusable full-image cache. Do not retain a
            # memory map into a job directory that the parent is about to close.
            denoise.report_phase('completed',eta_seconds=0.)
            response={'type':'result','status':snapshot()}
        except RenderCancelled:
            response={'type':'cancelled'}
        except Exception as exc:
            response={'type':'error','error':type(exc).__name__+': '+str(exc)}
        finally:
            if os.name=='nt':
                import ctypes as C
                C.windll.kernel32.SetPriorityClass(C.windll.kernel32.GetCurrentProcess(),0x20)
            denoise.clear_cache()
            mapping=getattr(source,'_mmap',None)
            if mapping is not None:mapping.close()
            source=None;clean=None
        # Terminal acknowledgement follows mmap release on every path, not
        # just success. The parent may now remove its job directory safely.
        response['job']=job;send(response)
    return 0

if __name__=='__main__':raise SystemExit(worker_main())
