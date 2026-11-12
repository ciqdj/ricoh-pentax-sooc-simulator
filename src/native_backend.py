"""Original DCU processing with explicit input routes and no colour fitting."""
from pathlib import Path
import json, math, os, re, struct, subprocess, sys, tempfile, time, threading
import numpy as np
import tifffile
from PIL import Image
from app_paths import ROOT
from photo_formats import (read_photo, srgb_encode, ICC, encode_png, resolve_format,
                           RAW_EXT, RASTER_EXT)
from native_container import write_tiff_rgb
from temp_storage import (TemporaryDirectory, temporary_directory, ensure_free_space,
                          native_required_bytes, native_worker_required_bytes,
                          subprocess_environment)

# White balance the engine applies when it develops a PEF, measured on a K-1 frame.
# A foreign sensor frame is divided by this before encoding so the engine's own
# balance lands on the source camera's balance instead of doubling up.
FOREIGN_TARGET_WB=(2.4844,1.0,1.7734)
LOOKS={key:{'name':name,'mode':mode} for key,name,mode in [
    ('Natural','自然',0),('Bright','鲜明',1),('Portrait','人像',16),
    ('Landscape','风景',17),('MIYABI','风雅',18),('Monochrome','黑白',19),
    ('Muted','柔和 · 低饱和',20),('Reversal','反转片',23),('BleachBypass','留银冲洗',24),
    ('Radiant','亮丽',25),('Flat','平面',27),('SATOBI','怀旧褪色',28),
    ('HARUBENI','春红',29),('KATEN','夏天',30),('KYUSHU','九秋',31),
    ('FUYUNO','冬野',32),('GOLD','金色',33)]}
SEASONS=('HARUBENI','KATEN','KYUSHU','FUYUNO')
SIZES={1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8,13:4}
# Exactly the previous sRGB encoding formula at every possible 16-bit input.
# This is input encoding, not a creative filter or an approximation of DCU.
_SRGB16=np.rint(np.clip(srgb_encode(np.arange(65536,dtype=np.float64)/65535),0,1)*65535).astype(np.uint16)


class RenderCancelled(Exception):
    pass


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():raise RenderCancelled()


def _native_dimensions(source):
    # PEF's full sensor page can be larger than both its embedded preview and
    # the eventual cropped render. Read geometry only; do not decode/rewrite RAW.
    with tifffile.TiffFile(source) as image:
        return max(((int(p.imagewidth),int(p.imagelength)) for p in image.pages),
                   key=lambda size:size[0]*size[1])


def _note_fields(note):
    if note[:10]!=b'PENTAX \0II':raise ValueError('Unsupported metadata layout')
    rows={}
    for i in range(struct.unpack_from('<H',note,10)[0]):
        at=12+i*12;tag,kind,count,value=struct.unpack_from('<HHII',note,at)
        n=SIZES[kind]*count;offset=at+8 if n<=4 else value
        if offset+n>len(note):raise ValueError('Invalid metadata offset')
        rows[tag]=(at,kind,count,note[offset:offset+n])
    return rows


def supplement_pef(source,target,bank):
    """Add same-camera mode tables to a copy; retain original encoded sensor data."""
    data=bytearray(Path(source).read_bytes())
    if data[:4]!=b'II*\0':raise ValueError('Unsupported PEF byte order')
    def entries(offset):
        n=struct.unpack_from('<H',data,offset)[0]
        return {struct.unpack_from('<H',data,offset+2+12*i)[0]:offset+2+12*i for i in range(n)}
    first=entries(struct.unpack_from('<I',data,4)[0])
    exif=entries(struct.unpack_from('<I',data,first[34665]+8)[0]);entry=exif[37500]
    _,kind,count,offset=struct.unpack_from('<HHII',data,entry)
    if kind!=7:raise ValueError('Unsupported PEF MakerNote type')
    note=bytearray(data[offset:offset+count]);rows=_note_fields(note);donor=_note_fields(bank)
    for tag in (0x21a,0x21b,0x404):
        if tag not in rows or tag not in donor:raise ValueError('Incomplete same-camera bank')
        at=rows[tag][0];_,kind,n,value=donor[tag]
        if len(note)%2:note.append(0)
        struct.pack_into('<HHII',note,at,tag,kind,n,len(note));note.extend(value)
    if len(data)%2:data.append(0)
    struct.pack_into('<II',data,entry+4,len(note),len(data));data.extend(note)
    target.write_bytes(data)


def _native(source,work,mode,ev,progress=None,cancelled=None,timeout=60,controls=None):
    ensure_free_space(work,native_worker_required_bytes(*_native_dimensions(source)))
    output=work/'native-output.tif'
    request=work/'request.json'
    request.write_text(json.dumps({'input':str(source),'output':str(output),
        'mode':mode,'exposure_ev':ev,'controls':controls or {}}),encoding='utf-8')
    if getattr(sys,'frozen',False):command=[sys.executable,'--native-worker',str(request)]
    else:command=[sys.executable,str(Path(__file__).with_name('native_worker.py')),str(request)]
    _check_cancel(cancelled)
    if progress:progress('加载原厂引擎')
    child=subprocess.Popen(command,cwd=work,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW,
        env=subprocess_environment(work))
    started=time.monotonic();last_stage=''
    try:
        while child.poll() is None:
            _check_cancel(cancelled)
            elapsed=time.monotonic()-started
            if elapsed>timeout:raise TimeoutError()
            stage_file=work/'stage.json'
            if progress and stage_file.is_file():
                try:stage=json.loads(stage_file.read_text(encoding='utf-8'))['stage']
                except (OSError,ValueError,KeyError):stage=''
                if stage and stage!=last_stage:progress(stage);last_stage=stage
            # AUDIT-PERF N-1: poll quickly while the native engine is still
            # loading, then back off. Only the progress label reads stage.json;
            # success is decided by the exit code plus result.json and the TIFF,
            # so the interval cannot change any pixel or acceptance rule.
            time.sleep(.05 if elapsed<1.0 else .15 if elapsed<5.0 else .35)
    except (RenderCancelled,TimeoutError) as exc:
        # Terminate only this task's worker process tree, including a bootloader.
        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        child.wait(timeout=15)
        if isinstance(exc,RenderCancelled):raise
        raise RuntimeError('原厂引擎处理超时，本次任务已停止；可以重新选择滤镜重试') from None
    if child.returncode or not output.is_file():
        log=(work/'worker.log').read_text(encoding='utf-8',errors='replace') if (work/'worker.log').exists() else ''
        # Do not expose native addresses or local filesystem paths in the UI.
        error=next((s for s in reversed(log.splitlines()) if s.startswith(('RuntimeError:','ValueError:','OSError:'))),'原厂引擎进程未正常完成')
        writer_error=re.search(r'Native TIFF writer failed \((-?\d+)\)',error)
        if writer_error:
            code=int(writer_error.group(1))
            failure=RuntimeError(f'原厂处理未能写入临时图像（错误码 {code}），请检查缓存磁盘空间和写入权限。')
            failure.native_error_code=code
            # Retain the exact native failure for internal traceback/diagnosis;
            # this status alone does not establish a disk-full cause.
            raise failure from RuntimeError(error)
        raise RuntimeError(error)
    if progress:progress('读取处理结果')
    _check_cancel(cancelled)
    array=tifffile.imread(output)
    if array.dtype!=np.uint16 or array.ndim!=3 or array.shape[2]!=3:
        raise RuntimeError('原厂引擎输出格式不符合 RGB 16 位要求')
    return array,json.loads((work/'result.json').read_text(encoding='utf-8'))


class RenderSession:
    """One photo's input preparation, reused by its selected look and cards.

    Input preparation is serialized; at most two independent native processes
    can read the immutable prepared files. Native parameters stay unchanged.
    """
    def __init__(self,source):
        self.source=Path(source).resolve()
        if not self.source.is_file():raise FileNotFoundError('找不到输入照片')
        self.rgb=None;self.alpha=None;self.notes=[];self.inputs={};self.source_model='';self.decode_count=0
        self.baselines={};self.baseline_lock=threading.Lock()
        self.prepare_lock=threading.Lock()
        self.temp=None;self.work=None;self._dimensions=None
        if self.source.suffix.lower()=='.pef':
            with tifffile.TiffFile(self.source) as image:
                tag=image.pages[0].tags.get(272)
                self.source_model=str(tag.value if tag else '').strip()
                self._dimensions=max(((int(p.imagewidth),int(p.imagelength)) for p in image.pages),
                                     key=lambda size:size[0]*size[1])
        elif self.source.suffix.lower() in ('.tif','.tiff'):
            from photo_formats import metadata_from
            self.source_model=metadata_from(self.source)[3].get('camera_model','')

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    @classmethod
    def from_rgb(cls,rgb,alpha=None,notes=(),source_model=''):
        """Use an immutable RGB16 working image without re-decoding a TIFF.

        This is explicitly the RGB compatibility route, never a synthetic RAW
        or a claim of camera sensor equivalence. Native parameter-bank files
        and every original processing call remain the same as file inputs.
        The caller retains the arrays for the lifetime of this session.
        """
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint16 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
            raise ValueError('原厂引擎内存输入需要 RGB16')
        if alpha is not None and (alpha.shape!=rgb.shape[:2] or not np.isfinite(alpha).all() or np.any(alpha<0) or np.any(alpha>1)):
            raise ValueError('原厂引擎透明通道无效')
        instance=cls.__new__(cls)
        instance.source=Path('rgb16-working-input.tif')
        instance.rgb=rgb;instance.alpha=alpha;instance.notes=list(notes)
        instance.inputs={};instance.source_model=str(source_model);instance.decode_count=0
        instance.baselines={};instance.baseline_lock=threading.Lock()
        instance.prepare_lock=threading.Lock();instance.temp=None;instance.work=None
        instance._dimensions=(rgb.shape[1],rgb.shape[0])
        return instance

    def close(self):
        self.rgb=self.alpha=None;self.baselines.clear()
        if self.temp is not None:self.temp.cleanup();self.temp=None;self.work=None

    def _workspace(self,width,height):
        # Called under prepare_lock, after decoding compatibility inputs. Two
        # render workers share immutable prepared inputs but have separate output.
        if self._dimensions is None or width*height>self._dimensions[0]*self._dimensions[1]:
            self._dimensions=(width,height)
        if self.temp is None:
            required=native_required_bytes(*self._dimensions,concurrent=2,
                source_bytes=self.source.stat().st_size if self.source.suffix.lower()=='.pef' else 0)
            self.temp=temporary_directory(required,prefix='PentaxNative-')
            self.work=Path(self.temp.name)

    def _prepare_space(self,input_bytes=0):
        ensure_free_space(self.work,int(input_bytes)+native_worker_required_bytes(*self._dimensions,concurrent=2))

    def rgb_input(self,progress=None,cancelled=None):
        if self.rgb is None:
            _check_cancel(cancelled)
            if progress:progress('解码 RAW 原片' if self.source.suffix.lower() in RAW_EXT else '读取照片')
            photo=read_photo(self.source);self.decode_count+=1
            _check_cancel(cancelled)
            if progress:progress('准备 16 位输入')
            rgb=photo.rgb
            if photo.linear:
                # Chunking keeps temporary allocations bounded and is bit-exact.
                for start in range(0,rgb.shape[0],128):
                    _check_cancel(cancelled);rgb[start:start+128]=_SRGB16[rgb[start:start+128]]
            elif rgb.dtype==np.uint8:rgb=rgb.astype(np.uint16)*257
            self.rgb=rgb;self.notes=list(photo.notes)
            self.alpha=photo.alpha.astype(np.float64)/np.iinfo(photo.alpha.dtype).max if photo.alpha is not None else None
        return self.rgb,self.alpha

    def _foreign_pef(self,progress,cancelled):
        """Return the source re-encoded once into the maker's raw container.

        AUDIT-P44: the conversion is keyed on the source file and cached for the
        whole machine, because it only depends on the decoded frame. Rebuilding it
        per render cost two seconds on every look change while producing the same
        bytes, and the look itself only needs the finished container.
        """
        suffix=self.source.suffix.lower()
        if suffix not in RAW_EXT or suffix=='.pef':return None
        skeleton=ROOT/'native-data/pef-skeleton.bin'
        if not skeleton.is_file():return None
        try:
            import hashlib
            from app_paths import writable_dir
            import pef_writer
            stat=self.source.stat()
            tag=hashlib.sha256(f'{self.source.resolve()}|{stat.st_size}|{int(stat.st_mtime)}'.encode('utf-8')).hexdigest()[:32]
            cache=writable_dir('foreign-pef')
            cached=cache/(tag+'.pef')
            if cached.is_file() and cached.stat().st_size>1024:
                if progress:progress('复用已转换的原厂容器')
                return cached
            _check_cancel(cancelled)
            if progress:progress('解码外来 RAW')
            frame,black,white,source_wb=pef_writer.bayer_from_raw(self.source)
            _check_cancel(cancelled)
            template=pef_writer.PefTemplate(skeleton)
            if progress:progress('适配原厂画幅')
            fitted=pef_writer.fit_to(frame,template.rows,template.columns,black)
            normalised=pef_writer.normalise(fitted,black,white,source_wb,FOREIGN_TARGET_WB)
            _check_cancel(cancelled)
            if progress:progress('封装原厂 RAW 容器')
            blob=template.encode(normalised,pef_writer._library())
            pending=cached.with_name(cached.name+'.writing')
            try:
                pending.write_bytes(blob)
                os.replace(pending,cached)
            finally:pending.unlink(missing_ok=True)
            return cached
        except Exception:
            return None
    def _input(self,look,progress,cancelled):
        _check_cancel(cancelled)
        mode=LOOKS[look]['mode']
        bank_id={'PENTAX K-1':'k1','PENTAX K-1 Mark II':'k1ii','PENTAX K-3 Mark III':'k3iii'}.get(self.source_model,'k1')
        if bank_id=='k3iii' and mode==29:bank_id='k1'
        bank=(ROOT/'native-data'/f'{bank_id}-parameters.bin').read_bytes()
        model={'k1':'PENTAX K-1','k1ii':'PENTAX K-1 Mark II','k3iii':'PENTAX K-3 Mark III'}[bank_id]
        # K-1 II's recovered bank contains only spring; use the complete K-1
        # RGB bank for its other seasons, never mislabel it as K-1 II RAW.
        if bank_id=='k1ii' and mode in (30,31,32):
            bank_id='k1';model='PENTAX K-1';bank=(ROOT/'native-data/k1-parameters.bin').read_bytes()
        native_input=self.source;route='native_pef';alpha=None;notes=[]
        if self.source.suffix.lower()=='.pef' and self.source_model==model:
            self._workspace(*self._dimensions)
            if mode in (29,30,31,32):
                native_input=self.work/(bank_id+'-parameters.pef')
                if not native_input.exists():
                    self._prepare_space(self.source.stat().st_size+len(bank)+65536)
                    if progress:progress('准备同机型原厂参数')
                    pending=native_input.with_name(native_input.name+'.writing')
                    try:
                        supplement_pef(self.source,pending,bank)
                        os.replace(pending,native_input)
                    finally:pending.unlink(missing_ok=True)
                route='native_pef_same_camera_bank'
        elif self.source.suffix.lower()=='.pef' and mode<29:self._workspace(*self._dimensions)
        else:
            # AUDIT-P42: a foreign RAW is re-encoded into the maker's own raw
            # container instead of being handed over as finished RGB. The RGB
            # route skipped the develop half of the engine, so another brand
            # never received the tone curve the camera look is built on.
            rgb,alpha=self.rgb_input(progress,cancelled)
            self._workspace(rgb.shape[1],rgb.shape[0])
            pending_pef=self._foreign_pef(progress,cancelled)
            native_input=self.work/(bank_id+'-foreign.pef')
            if pending_pef is not None:
                if 'foreign' not in self.inputs:
                    # AUDIT-P44: the copied container carries the mode tables, so it
                    # is as reusable as the frame it came from. Keeping it next to
                    # the frame cache avoids copying tens of megabytes per look.
                    source_stat=self.source.stat()
                    cache_key=f'{pending_pef.stem}-{bank_id}-{source_stat.st_size}-{int(source_stat.st_mtime)}.pef'
                    from app_paths import writable_dir
                    canned=writable_dir('foreign-pef')/cache_key
                    if canned.is_file() and canned.stat().st_size>1024:
                        if progress:progress('复用已准备的原厂容器')
                        native_input=canned
                    else:
                        self._prepare_space(pending_pef.stat().st_size+len(bank)+65536)
                        if progress:progress('准备原厂引擎输入')
                        staged=native_input.with_name(native_input.name+'.writing')
                        try:
                            supplement_pef(pending_pef,staged,bank)
                            os.replace(staged,native_input)
                        finally:staged.unlink(missing_ok=True)
                        try:
                            import shutil
                            shutil.copyfile(native_input,canned.with_name(canned.name+'.writing'))
                            os.replace(canned.with_name(canned.name+'.writing'),canned)
                        except Exception:
                            canned.with_name(canned.name+'.writing').unlink(missing_ok=True)
                    self.inputs['foreign']=native_input
                route='native_pef_foreign';notes=list(self.notes)
                notes.append('Foreign RAW re-encoded into the maker raw container; developed by the original engine.')
            else:
                native_input=self.work/(bank_id+'-rgb.tif')
                if bank_id not in self.inputs:
                    self._prepare_space(rgb.nbytes+len(bank)+65536)
                    if progress:progress('准备原厂引擎输入')
                    pending=native_input.with_name(native_input.name+'.writing')
                    try:
                        write_tiff_rgb(pending,rgb,bank,model_name=model)
                        os.replace(pending,native_input)
                    finally:pending.unlink(missing_ok=True)
                    self.inputs[bank_id]=native_input
                route='rgb_compatibility';notes=list(self.notes)
                notes.append('RGB compatibility input; camera RAW pipeline equivalence is not verified.')
        _check_cancel(cancelled)
        return native_input,route,alpha,notes,model

    def baseline(self,look='Natural',progress=None,cancelled=None):
        with self.prepare_lock:
            native_input,route,alpha,notes,model=self._input(look,progress,cancelled)
            if route=='rgb_compatibility':return self.rgb,self.alpha
        # Original PEF as-shot processing gives identical geometry, without
        # resizing an embedded JPEG or guessing sensor crop margins.
        with self.baseline_lock:
            key=str(self.source)
            if key not in self.baselines:
                with TemporaryDirectory(prefix='baseline-',dir=self.work) as folder:
                    rgb,_=_native(self.source,Path(folder),-1,0,progress,cancelled)
                self.baselines[key]=rgb
            return self.baselines[key],None

    def render(self,look='Natural',exposure=0.0,progress=None,cancelled=None,controls=None):
        if look not in LOOKS:raise ValueError('未知原厂模式')
        if not math.isfinite(exposure) or not -5<=exposure<=5:raise ValueError('曝光范围为 −5 至 +5 EV')
        with self.prepare_lock:
            native_input,route,alpha,notes,model=self._input(look,progress,cancelled)
        with TemporaryDirectory(prefix='render-',dir=self.work) as folder:
            array,report=_native(native_input,Path(folder),LOOKS[look]['mode'],exposure,progress,cancelled,controls=controls)
        report.update({'look':look,'input_route':route,'parameter_bank_model':model if route!='native_pef' else self.source_model,
            'notes':notes,'native_width':array.shape[1],'native_height':array.shape[0]})
        return array,alpha,report

    def render_noise_comparison(self,look='Natural',exposure=0.0,progress=None,
                                cancelled=None,controls=None):
        """Compare original-default and explicit NR at the same colour settings.

        Both images run the complete original development chain. The first is
        NOT an NR-disabled image: omitting custom NR preserves the original
        input/default settings. This API intentionally has no filter-amount
        blend; using the undecoded/unprocessed input as either endpoint would
        conflate colour changes with noise reduction.
        """
        from native_controls import NR_CONTROLS, validate_overrides
        if look not in LOOKS:raise ValueError('未知原厂模式')
        effective={} if controls is None else controls
        validate_overrides(LOOKS[look]['mode'],effective)
        effective=dict(effective)
        before_controls={key:value for key,value in effective.items()
                         if key not in NR_CONTROLS}
        before,alpha,before_report=self.render(look,exposure,progress,cancelled,
                                              before_controls)
        _check_cancel(cancelled)
        if before_controls==effective:
            after,after_report=before,before_report
        else:
            after,after_alpha,after_report=self.render(look,exposure,progress,
                                                       cancelled,effective)
            if after.shape!=before.shape:
                raise RuntimeError('原厂降噪对照的图像尺寸不一致')
            if ((alpha is None)!=(after_alpha is None) or
                    (alpha is not None and not np.array_equal(alpha,after_alpha))):
                raise RuntimeError('原厂降噪对照的透明度不一致')
        report={'comparison_kind':'factory_nr_vs_custom_nr',
                'same_colour_settings':True,'same_exposure':True,
                'filter_amount':1.0,'before_is_nr_disabled':False,
                'before':before_report,'after':after_report}
        return before,after,alpha,report


def render(source,look='Natural',exposure=0.0,controls=None):
    with RenderSession(source) as session:return session.render(look,exposure,controls=controls)


def mix_amount(base,result,amount):
    if amount==1:return result
    if not math.isfinite(amount) or not 0<=amount<=2:raise ValueError('浓淡范围为 0–200%')
    if base.shape!=result.shape:raise ValueError('输入和输出尺寸不同，不能混合浓淡')
    if amount==0:return base.copy()
    out=np.empty_like(result)
    for row in range(0,result.shape[0],128):
        a=base[row:row+128].astype(np.float64);b=result[row:row+128].astype(np.float64)
        out[row:row+128]=np.rint(np.clip(a+(b-a)*amount,0,65535)).astype(np.uint16)
    return out


def save(source,destination,look='Natural',exposure=0.0,jpeg_quality=98,bit_depth=16,controls=None,amount=1.0,progress=None,cancelled=None,post_balance=(0,0),post_settings=None):
    source=Path(source).resolve();destination=Path(destination).resolve()
    sidecar=destination.with_name(destination.name+'.json')
    if source==destination or destination.exists() or sidecar.exists():raise ValueError('输出必须是新文件，不覆盖原片')
    if destination.suffix.lower() not in RASTER_EXT:raise ValueError('不支持的输出格式')
    if bit_depth not in (8,16) or not 80<=jpeg_quality<=100:raise ValueError('导出设置无效')
    fmt=resolve_format(destination)
    with RenderSession(source) as session:
        rgb,alpha,report=session.render(look,exposure,progress,cancelled,controls)
        if amount!=1:rgb=mix_amount(session.baseline(look,progress,cancelled)[0],rgb,amount)
    report['user_amount_percent']=round(amount*100,2)
    report['user_amount_mix_applied']=amount!=1
    from postprocess import process as post_process,annotate
    rgb=post_process(rgb,post_balance,post_settings,cancelled,progress,scale=1.0,alpha=alpha)
    annotate(report,post_balance,post_settings)
    bits=bit_depth if fmt in ('png','tiff') else 8
    keep_alpha=alpha is not None and fmt in ('png','tiff','webp')
    if alpha is not None and not keep_alpha:
        rgb=np.rint(rgb.astype(np.float64)*alpha[...,None]+65535*(1-alpha[...,None])).astype(np.uint16)
    if bits==8:rgb=((rgb.astype(np.uint32)+128)//257).astype(np.uint8)
    if keep_alpha:
        a=np.rint(alpha*(65535 if bits==16 else 255)).astype(rgb.dtype)
        rgb=np.concatenate([rgb,a[...,None]],axis=2)
    from photo_formats import capture_for_output,save_raster
    exif,capture=capture_for_output(source,rgb.shape)
    destination.parent.mkdir(parents=True,exist_ok=True);created=False;sidecar_created=False
    try:
        save_raster(destination,rgb,fmt,jpeg_quality,keep_alpha,exif);created=True
        report.update({'output_format':fmt,'output_bits':bits,'lossless_encoding':fmt!='jpeg',
            'alpha_retained':keep_alpha,'capture_metadata':capture,'personal_metadata_copied':False})
        with sidecar.open('x',encoding='utf-8') as out:
            sidecar_created=True;json.dump(report,out,ensure_ascii=False,indent=2)
        return report
    except Exception:
        if created:destination.unlink(missing_ok=True)
        if sidecar_created:sidecar.unlink(missing_ok=True)
        raise
