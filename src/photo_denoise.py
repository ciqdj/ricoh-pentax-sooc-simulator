"""Photo-scoped input denoising, independent of every downstream colour look.

No Tk objects or callbacks are retained. request/status/get are non-decoding
calls; prepare/get_or_compute/working_input belong on a background worker.
Keep the returned Result alive while its TIFF path is used by a renderer.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass,field
from pathlib import Path
import math,os,re,stat,threading,time,uuid,weakref
import numpy as np
from temp_storage import temporary_directory


@dataclass(frozen=True)
class PhotoIdentity:
    path:str
    size:int
    mtime_ns:int
    ctime_ns:int
    device:int
    inode:int


@dataclass(frozen=True)
class PhotoKey:
    identity:PhotoIdentity
    model:str
    backend_mode:str


def identity_for(path):
    path=Path(path).resolve(strict=True);value=path.stat()
    if not stat.S_ISREG(value.st_mode):raise ValueError('请选择一个照片文件')
    return PhotoIdentity(os.path.normcase(str(path)),value.st_size,value.st_mtime_ns,
                         value.st_ctime_ns,value.st_dev,value.st_ino)


def key_for(path,model='nafnet',backend_mode='auto'):
    from denoise import MODELS
    if model not in MODELS:raise ValueError('不支持的降噪模型')
    if backend_mode not in ('auto','cpu'):raise ValueError('不支持的硬件模式')
    return PhotoKey(identity_for(path),model,'cpu' if model=='dcu' else backend_mode)


class SourceChanged(RuntimeError):
    pass


def _current(identity):
    try:return identity_for(identity.path)==identity
    except (OSError,ValueError):return False


def _assert_current(identity):
    if not _current(identity):raise SourceChanged('原片已改变或无法访问，请重新准备照片')


def _cancelled(callback):
    if callback and callback():
        from native_backend import RenderCancelled
        raise RenderCancelled()


@dataclass(frozen=True)
class PreparedPhoto:
    identity:PhotoIdentity
    rgb:np.ndarray
    alpha:object
    notes:tuple
    seconds:float
    linear_rgb:object=None
    exif:bytes=b''
    metadata:dict|None=None

    @property
    def size(self):return self.rgb.shape[1],self.rgb.shape[0]


class PhotoDenoiseResult:
    def __init__(self,key,prepared,rgb,path,storage,seconds,backend,timings):
        self.key=key;self.prepared=prepared;self.source=prepared.rgb
        self.rgb=rgb;self.clean=rgb;self.alpha=prepared.alpha
        self.run_id=uuid.uuid4().hex
        self.path=Path(path);self.seconds=seconds;self.backend=dict(backend)
        self.notes=prepared.notes;self.timings=dict(timings);self.size=prepared.size
        # No explicit cache eviction closes a held Result's file. External
        # references (including a render context) keep this owner alive.
        self._cleanup=weakref.finalize(self,storage.cleanup)


@dataclass
class _Task:
    key:object
    task_id:str=field(default_factory=lambda:uuid.uuid4().hex)
    submitted_at:float=field(default_factory=time.monotonic)
    cancel:threading.Event=field(default_factory=threading.Event)
    future:object=None
    state:str='queued'
    stage:str='等待照片降噪任务…'
    percent:object=None
    started:float=0.
    ended:float=0.
    error:str=''
    error_code:str=''
    last_stat:float=0.
    phase:str='queued'
    done:object=None
    total:object=None
    eta_seconds:object=None
    backend:object=None
    cancel_reason:str=''
    protected_consumers:int=0
    supersede_deferred:bool=False


_ENCODE16=None


def _decode(identity,cancelled,progress):
    global _ENCODE16
    from photo_formats import read_photo,srgb_encode
    started=time.monotonic();_assert_current(identity);_cancelled(cancelled)
    progress('读取原始照片…')
    photo=read_photo(identity.path,half_size=False)
    _cancelled(cancelled);_assert_current(identity)
    progress('准备原尺寸 sRGB16 输入…')
    if photo.rgb.dtype==np.uint8:
        rgb=np.multiply(photo.rgb,257,dtype=np.uint16)
    elif photo.rgb.dtype==np.uint16:rgb=photo.rgb
    else:raise ValueError('输入照片需要 RGB8 或 RGB16')
    linear_rgb=None
    if photo.linear:
        if _ENCODE16 is None:
            _ENCODE16=np.rint(np.clip(srgb_encode(np.arange(65536,dtype=np.float64)/65535),0,1)*65535).astype(np.uint16)
        # Keep the decoder's linear values for the direct GR input route.
        # AI models and existing manual edits use their own encoded RGB16.
        linear_rgb=rgb;linear_rgb.flags.writeable=False;rgb=np.empty_like(linear_rgb)
        for top in range(0,len(rgb),128):
            _cancelled(cancelled);rgb[top:top+128]=_ENCODE16[linear_rgb[top:top+128]]
    alpha=photo.alpha
    if alpha is not None:
        if alpha.dtype==np.uint8:alpha=np.multiply(alpha,257,dtype=np.uint16)
        elif alpha.dtype!=np.uint16:raise ValueError('不支持的透明通道位深')
        alpha.flags.writeable=False
    rgb.flags.writeable=False
    return PreparedPhoto(identity,rgb,alpha,tuple(photo.notes),time.monotonic()-started,
                         linear_rgb,photo.exif,photo.metadata)


def _write_tiff(path,rgb,alpha,cancelled=None,clean=None,amount=1.,exif=b''):
    """Write one standalone TIFF without a second full RGBA/blend array."""
    import tifffile
    from photo_formats import ICC
    if Path(path).exists():raise ValueError('工作副本不能覆盖已有文件')
    shape=(*rgb.shape[:2],4 if alpha is not None else 3)
    target=tifffile.memmap(path,shape=shape,dtype=np.uint16,photometric='rgb',
        extrasamples='unassalpha' if alpha is not None else None,iccprofile=ICC,
        metadata=None,software='')
    try:
        for top in range(0,len(rgb),128):
            _cancelled(cancelled);bottom=top+128
            if clean is not None:
                from postprocess import blend_clean
                target[top:bottom,:,:3]=blend_clean(rgb[top:bottom],clean[top:bottom],amount,cancelled)
            else:target[top:bottom,:,:3]=rgb[top:bottom]
            if alpha is not None:target[top:bottom,:,3]=alpha[top:bottom]
        target.flush();_cancelled(cancelled)
    finally:target._mmap.close()
    if exif:
        from PIL import Image
        from photo_formats import write_tiff_exif
        parsed=Image.Exif();parsed.load(exif)
        if not write_tiff_exif(path,parsed):raise RuntimeError('降噪工作副本的拍摄参数写入失败')


@contextmanager
def working_input(result,amount_percent=100,cancelled=None):
    """Lease a full-size input TIFF for one rendering/export generation.

    100% reuses the completed TIFF. Other strengths use their own temporary
    directory for exactly this context, never an unbounded strength cache.
    """
    amount=float(amount_percent)
    if not math.isfinite(amount) or not 0<=amount<=100:raise ValueError('降噪混合比例需要在 0–100 之间')
    _cancelled(cancelled);_assert_current(result.key.identity)
    if amount==100:
        yield result.path
        return
    required=result.source.nbytes+(0 if result.alpha is None else result.alpha.nbytes)+64*1024*1024
    with temporary_directory(required,prefix='photo-input-blend-') as folder:
        path=Path(folder)/'input.tif'
        _write_tiff(path,result.source,result.alpha,cancelled,
                    result.rgb if amount else None,amount/100,result.prepared.exif)
        _cancelled(cancelled);_assert_current(result.key.identity)
        yield path


class PhotoDenoiseStore:
    """One active operation, bounded queue and LRU photo/model result cache.

    max_bytes is a retained-array target: an active single large photograph
    is kept at original size even when it exceeds this target. Other inactive
    entries are evicted. A caller-held Result may outlive the store/cache.
    """
    key_for=staticmethod(key_for)

    def __init__(self,max_entries=2,max_bytes=512*1024*1024,max_pending=8):
        if max_entries<1 or max_bytes<1 or max_pending<1:raise ValueError('缓存限制必须为正数')
        self.max_entries=int(max_entries);self.max_bytes=int(max_bytes);self.max_pending=int(max_pending)
        self._lock=threading.RLock();self._changed=threading.Condition(self._lock)
        self._pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='PhotoInputDenoise')
        self._tasks=OrderedDict();self._history=OrderedDict()
        self._sources=OrderedDict();self._results=OrderedDict();self._previews=OrderedDict()
        self._active=None;self._active_task=None;self._selected=None;self._closed=False

    @staticmethod
    def _identity(key):return key.identity if isinstance(key,PhotoKey) else key

    def _get_cached(self,key):
        if isinstance(key,PhotoKey):return self._results.get(key)
        return self._sources.get(key)

    def _cancel_task(self,task,reason='user'):
        """Called under the condition lock; running work acknowledges its own exit."""
        if task.state not in ('queued','preparing','running'):return
        if reason=='superseded' and task.protected_consumers:
            # A captured export is an explicit consumer of this exact model.
            # UI model changes may wait for it, but cannot invalidate its work.
            task.supersede_deferred=True;return
        task.cancel.set();task.cancel_reason=reason
        task.stage='正在停止被替换的模型…' if reason=='superseded' else '正在取消…'
        task.percent=None;task.eta_seconds=None
        if task.future is not None and task.future.cancel():
            task.state='cancelled';task.stage='已被新模型替换' if reason=='superseded' else '已取消'
            task.phase='cancelled';task.ended=time.monotonic()

    def _prune_tasks(self):
        # Active/cancelling tasks remain visible even if a new request with
        # the same PhotoKey has replaced the key's current-task mapping.
        for task_id,task in tuple(self._history.items()):
            if len(self._history)<=64:break
            if task.future is not None and task.future.done():
                self._history.pop(task_id,None)
                if self._tasks.get(task.key) is task:self._tasks.pop(task.key,None)

    def _request(self,key,supersede=True):
        with self._changed:
            if self._closed:raise RuntimeError('照片降噪任务已关闭')
            if isinstance(key,PhotoKey):
                self._selected=key
                if supersede:
                    for task in self._history.values():
                        if isinstance(task.key,PhotoKey) and task.key.identity==key.identity and task.key!=key:
                            self._cancel_task(task,'superseded')
                    self._changed.notify_all()
            if self._get_cached(key) is not None:return key
            previous=self._tasks.get(key)
            if previous is not None and previous.future is not None and not previous.future.done() and not previous.cancel.is_set():return key
            if sum(t.state=='queued' for t in self._history.values())>=self.max_pending:
                raise RuntimeError('待处理照片较多，请稍后再试')
            task=_Task(key);self._tasks[key]=task;self._history[task.task_id]=task
            task.future=self._pool.submit(self._execute,task)
            self._prune_tasks()
            self._changed.notify_all()
            return key

    def request(self,path,model='nafnet',backend_mode='auto',*,supersede=True):
        return self._request(key_for(path,model,backend_mode),supersede=supersede)

    def request_prepare(self,path):
        identity=path if isinstance(path,PhotoIdentity) else identity_for(path)
        _assert_current(identity)
        return self._request(identity,supersede=False)

    def get_source(self,path):
        try:identity=path if isinstance(path,PhotoIdentity) else identity_for(path)
        except (OSError,ValueError):return None
        if not _current(identity):return None
        with self._lock:
            prepared=self._sources.get(identity)
            if prepared is not None:self._sources.move_to_end(identity)
            return prepared

    def get(self,key):
        if not isinstance(key,PhotoKey) or not _current(key.identity):return None
        with self._lock:
            result=self._results.get(key)
            if result is not None:self._results.move_to_end(key)
            return result

    def get_preview(self,key):
        if not isinstance(key,PhotoKey) or not _current(key.identity):return None
        with self._lock:
            result=self._results.get(key);preview=self._previews.get(key)
            if result is not None:
                h,w=result.rgb.shape[:2];ww=min(256,w);hh=min(256,h);x=(w-ww)//2;y=(h-hh)//2
                return {'source':result.source[y:y+hh,x:x+ww], 'clean':result.rgb[y:y+hh,x:x+ww],
                        'bounds':(x,y,ww,hh),'center':(.5,.5),'full':True}
            return preview

    def discard(self,key):
        """Forget one result and its preview without invalidating in-use readers."""
        if not isinstance(key,PhotoKey):raise ValueError('只能丢弃照片降噪结果')
        with self._changed:
            task=self._tasks.pop(key,None)
            if task is not None:self._cancel_task(task,'discarded')
            result=self._results.pop(key,None)
            preview=self._previews.pop(key,None)
            if self._selected==key:self._selected=None
            self._changed.notify_all()
        # A render/export holding Result keeps its TIFF alive until it exits.
        return result is not None or preview is not None or task is not None

    @staticmethod
    def _task_status(task,now):
        end=task.ended or now
        wait_end=task.started or end
        return {'task_id':task.task_id,'state':task.state,'stage':task.stage,
                'phase':task.phase,'percent':task.percent,'submitted_at':task.submitted_at,
                'started':task.started or None,'elapsed':max(0.,end-task.submitted_at),
                'wait_elapsed':max(0.,wait_end-task.submitted_at),
                'run_elapsed':max(0.,end-task.started) if task.started else 0.,
                'done':task.done,'total':task.total,'eta_seconds':task.eta_seconds,
                'backend':task.backend,'cancel_requested':task.cancel.is_set(),
                'cancel_reason':task.cancel_reason,'protected_consumers':task.protected_consumers,
                'supersede_deferred':task.supersede_deferred,'error':task.error,'error_code':task.error_code}

    def _queue_details(self):
        active=self._active_task
        details={}
        if active is not None:
            details[active.task_id]={'queue_position':0,'blocked_by':None,'wait_eta_seconds':0.}
        previous=active.task_id if active is not None else None
        # Only the first queued job can inherit a measured running ETA. The
        # duration of another, not-yet-run image/model is unknown; do not make
        # up a total completion estimate for the rest of the queue.
        wait_eta=active.eta_seconds if active is not None else None
        if active is not None and active.cancel.is_set():wait_eta=None
        queued=(task for task in self._history.values() if task.state=='queued')
        for position,task in enumerate(queued,1):
            details[task.task_id]={'queue_position':position,'blocked_by':previous,
                                   'wait_eta_seconds':wait_eta}
            previous=task.task_id;wait_eta=None
        return details

    def status(self,key=None):
        with self._lock:
            key=key if key is not None else (self._active or self._selected)
            task=self._tasks.get(key);cached=self._get_cached(key)
            value={'key':key,'task_id':None,'state':'missing','stage':'','phase':'missing',
                   'percent':None,'submitted_at':None,'started':None,'elapsed':0.,
                   'wait_elapsed':0.,'run_elapsed':0.,'done':None,'total':None,
                   'eta_seconds':None,'backend':None,'queue_position':None,'blocked_by':None,
                   'wait_eta_seconds':None,'cancel_requested':False,'cancel_reason':'',
                   'protected_consumers':0,'supersede_deferred':False,
                   'error':'','error_code':'','result_path':None}
            if task is not None:
                value.update(self._task_status(task,time.monotonic()))
                value.update(self._queue_details().get(task.task_id,{}))
            if cached is not None:
                value.update(state='completed',stage='原始输入降噪已完成' if isinstance(key,PhotoKey) else '原始输入已准备',
                             phase='completed',percent=100,eta_seconds=0.,cancel_requested=False,
                             queue_position=None,blocked_by=None,wait_eta_seconds=None,
                             result_path=cached.path if isinstance(key,PhotoKey) else None)
                if task is None:value.update(elapsed=cached.seconds,run_elapsed=cached.seconds)
                if isinstance(key,PhotoKey) and value['backend'] is None:value['backend']=cached.backend.get('backend')
            elif value['state']=='completed':value.update(state='missing',phase='missing',stage='工作副本已移出缓存',percent=None,eta_seconds=None)
        if key is not None and not _current(self._identity(key)):
            value.update(state='failed',phase='failed',stage='原片已改变，请重新准备',percent=None,eta_seconds=None,
                         error='原片已改变或无法访问',error_code='source_changed',result_path=None)
        return value

    def tasks_snapshot(self,limit=16):
        """Small UI-safe task list: no PhotoKey, full paths or raw exceptions."""
        if limit<1:raise ValueError('任务列表限制必须为正数')
        with self._lock:
            pending=[task for task in self._history.values() if task.state in ('queued','preparing','running')]
            finished=[task for task in self._history.values() if task.state not in ('queued','preparing','running')]
            tasks=pending+finished[-max(0,int(limit)-len(pending)):] if len(pending)<limit else pending
            queue=self._queue_details();now=time.monotonic();rows=[]
            for task in tasks:
                row=self._task_status(task,now);identity=self._identity(task.key)
                row.pop('error',None)
                row.update(name=Path(identity.path).name,
                           model=task.key.model if isinstance(task.key,PhotoKey) else None,
                           backend_mode=task.key.backend_mode if isinstance(task.key,PhotoKey) else None,
                           queue_position=None,blocked_by=None,wait_eta_seconds=None)
                row.update(queue.get(task.task_id,{}))
                if task.state=='failed':row['stage']='原片已改变，请重新准备' if task.error_code=='source_changed' else '处理失败'
                else:
                    row['stage']=row['stage'].replace(identity.path,row['name']).replace(identity.path.replace('\\','/'),row['name'])
                rows.append(row)
            return rows

    def _cancelled(self,task):
        if self._closed or task.cancel.is_set():return True
        now=time.monotonic()
        if now-task.last_stat>=.25:
            task.last_stat=now
            if not _current(self._identity(task.key)):
                task.error_code='source_changed';task.cancel.set();return True
        return False

    def _progress(self,task,text,state=None,phase=None):
        with self._changed:
            if self._tasks.get(task.key) is not task or task.cancel.is_set():return
            text=str(text);match=re.search(r'(\d+(?:\.\d+)?)%',text)
            percent=float(match.group(1)) if match else None
            if percent is not None and percent>=100:text='正在准备降噪工作副本…';percent=None
            task.stage=text;task.percent=percent
            if state is not None:task.state=state
            if phase is not None:
                task.phase=phase;task.done=None;task.total=None;task.eta_seconds=None
            self._changed.notify_all()

    def _worker_status(self,task,value,preview=False):
        """Consume the worker's structured values without estimating from text."""
        if not isinstance(value,dict):return
        def number(name,integer=False):
            item=value.get(name)
            if isinstance(item,bool) or not isinstance(item,(int,float)) or not math.isfinite(item) or item<0:return None
            return int(item) if integer else float(item)
        with self._changed:
            if self._tasks.get(task.key) is not task or task.cancel.is_set():return
            phase=value.get('phase')
            if isinstance(phase,str) and phase:
                # Inference completion is not task completion: the full TIFF
                # still has to be read back and atomically published.
                task.phase=('preview_'+phase) if preview else ('receiving_result' if phase=='completed' else phase)
            backend=value.get('backend')
            if isinstance(backend,str):task.backend=backend
            if phase=='processing_tiles':
                task.done=number('done',True)
                if task.done is None:task.done=number('tiles_done',True)
                task.total=number('total',True)
                if task.total is None:task.total=number('tiles_total',True)
                if task.total is not None and (task.total<1 or task.done is not None and task.done>task.total):
                    task.done=None;task.total=None
                task.eta_seconds=None if preview else number('eta_seconds')
            else:task.done=None;task.total=None;task.eta_seconds=None
            self._changed.notify_all()

    def _bytes(self):
        arrays={}
        def retain(array):
            if array is None:return
            # Count the owning allocation once, including arrays kept by a
            # Result after its PreparedPhoto left the source LRU.
            while isinstance(array.base,np.ndarray):array=array.base
            arrays[id(array)]=array
        def prepared(photo):
            for array in (photo.rgb,photo.alpha,photo.linear_rgb):retain(array)
        for photo in self._sources.values():
            prepared(photo)
        for result in self._results.values():
            prepared(result.prepared)
            for array in (result.source,result.rgb,result.alpha):retain(array)
        for preview in self._previews.values():
            for array in (preview['source'],preview['clean']):retain(array)
        return sum(array.nbytes for array in arrays.values())

    def _trim(self,protect=None):
        # Called only from the task worker; dropping the last file owner must
        # not turn a main-thread get/status call into filesystem cleanup.
        protected={key for key in (protect,self._active,self._selected) if key is not None}
        dropped=[]
        while len(self._previews)>self.max_entries:
            key=next((key for key in self._previews if key not in protected),None)
            if key is None:break
            dropped.append(self._previews.pop(key))
        while len(self._results)>self.max_entries or self._bytes()>self.max_bytes:
            key=next((key for key in self._results if key not in protected),None)
            if key is None:break
            dropped.append(self._results.pop(key))
            self._previews.pop(key,None)
        needed={result.key.identity for result in self._results.values()}
        needed.update(self._identity(key) for key in protected)
        while len(self._sources)>self.max_entries or self._bytes()>self.max_bytes:
            key=next((key for key in self._sources if key not in needed),None)
            if key is None:break
            dropped.append(self._sources.pop(key))
        return dropped

    def _prepare(self,identity,task):
        with self._lock:prepared=self._sources.get(identity)
        if prepared is not None:return prepared
        prepared=_decode(identity,lambda:self._cancelled(task),lambda text:self._progress(task,text,'preparing','preparing_input'))
        _cancelled(lambda:self._cancelled(task));_assert_current(identity)
        with self._changed:
            if self._closed:_cancelled(lambda:True)
            self._sources[identity]=prepared;self._sources.move_to_end(identity)
            dropped=self._trim(task.key);self._changed.notify_all()
        dropped.clear()
        return prepared

    def _preview(self,key,prepared,task,infer):
        """Early native-pixel centre sample; full inference remains unchanged."""
        from denoise import TILING
        h,w=prepared.rgb.shape[:2];ww=min(256,w);hh=min(256,h);x=(w-ww)//2;y=(h-hh)//2
        margin=TILING[key.model][1]
        x0=max(0,((x-margin)//8)*8);y0=max(0,((y-margin)//8)*8)
        x1=min(w,((x+ww+margin+7)//8)*8);y1=min(h,((y+hh+margin+7)//8)*8)
        # If context is the entire source, full inference itself is already
        # small. Run it once and publish its centre as the completed preview.
        if (x0,y0,x1,y1)==(0,0,w,h):return
        cancel=lambda:self._cancelled(task)
        def progress(text):
            text='正在试算原尺寸局部预览…' if '%' in str(text) else '局部预览 · '+str(text)
            self._progress(task,text,'running')
        self._progress(task,'准备原尺寸局部预览…','running','preparing_preview')
        clean,_=infer(prepared.rgb[y0:y1,x0:x1],key.model,key.backend_mode,cancel,progress,
                      on_status=lambda value:self._worker_status(task,value,preview=True))
        _cancelled(cancel)
        before=prepared.rgb[y:y+hh,x:x+ww].copy()
        after=clean[y-y0:y-y0+hh,x-x0:x-x0+ww].copy()
        before.flags.writeable=False;after.flags.writeable=False
        preview={'source':before,'clean':after,'bounds':(x,y,ww,hh),'center':(.5,.5),'full':False}
        with self._changed:
            _cancelled(cancel)
            self._previews[key]=preview;self._previews.move_to_end(key)
            dropped=self._trim(key);self._changed.notify_all()
        dropped.clear()

    def _compute(self,key,task):
        started=time.monotonic();prepared=self._prepare(key.identity,task)
        cancel=lambda:self._cancelled(task)
        progress=lambda text:self._progress(task,text,'running')
        _cancelled(cancel);infer_started=time.monotonic()
        if key.model=='dcu':
            from dcu_denoise import apply
            rgb=apply(prepared.rgb,cancel,progress)
            backend={'model':'dcu','backend':'CPU · DCU','native_module_defaults':True}
        else:
            from neural_process import infer
            self._preview(key,prepared,task,infer)
            _cancelled(cancel);self._progress(task,'开始处理整张照片…','running','preparing_inference')
            rgb,backend=infer(prepared.rgb,key.model,key.backend_mode,cancel,progress,
                              on_status=lambda value:self._worker_status(task,value))
        infer_seconds=time.monotonic()-infer_started
        if rgb.dtype!=np.uint16 or rgb.shape!=prepared.rgb.shape:raise RuntimeError('降噪结果尺寸或位深不匹配')
        rgb.flags.writeable=False;_cancelled(cancel);_assert_current(key.identity)
        self._progress(task,'保存原尺寸 TIFF16 工作副本…','preparing','writing_working_copy')
        required=rgb.nbytes+(0 if prepared.alpha is None else prepared.alpha.nbytes)+64*1024*1024
        storage=temporary_directory(required,prefix='photo-input-denoise-')
        try:
            path=Path(storage.name)/'denoised-input.tif';pending=Path(storage.name)/'pending.tif';write_started=time.monotonic()
            _write_tiff(pending,rgb,prepared.alpha,cancel,exif=prepared.exif)
            _cancelled(cancel);_assert_current(key.identity)
            os.replace(pending,path)
            return PhotoDenoiseResult(key,prepared,rgb,path,storage,time.monotonic()-started,backend,
                {'denoise_seconds':infer_seconds,'write_seconds':time.monotonic()-write_started})
        except BaseException:storage.cleanup();raise

    def _execute(self,task):
        from native_backend import RenderCancelled
        try:
            with self._changed:
                self._active=task.key;self._active_task=task
                task.started=time.monotonic();task.state='preparing';task.phase='preparing_input'
                self._changed.notify_all()
            _cancelled(lambda:self._cancelled(task));_assert_current(self._identity(task.key))
            if isinstance(task.key,PhotoKey):
                result=self._compute(task.key,task)
                with self._changed:
                    _cancelled(lambda:self._cancelled(task))
                    self._results[task.key]=result;self._results.move_to_end(task.key);dropped=self._trim(task.key)
                dropped.clear()
            else:self._prepare(task.key,task)
            with self._changed:
                task.state='completed';task.percent=100;task.stage='已完成';task.phase='completed';task.eta_seconds=0.
        except (RenderCancelled,SourceChanged) as exc:
            changed=isinstance(exc,SourceChanged) or task.error_code=='source_changed'
            with self._changed:
                task.state='failed' if changed else 'cancelled';task.percent=None
                task.stage=task.error='原片已改变，请重新准备' if changed else ('已被新模型替换' if task.cancel_reason=='superseded' else '已取消')
                task.error_code='source_changed' if changed else ''
                task.phase='failed' if changed else 'cancelled';task.eta_seconds=None
        except Exception as exc:
            changed=not _current(self._identity(task.key))
            with self._changed:
                task.state='failed';task.phase='failed';task.percent=None;task.eta_seconds=None
                task.error_code='source_changed' if changed else ''
                task.error=task.stage='原片已改变，请重新准备' if changed else type(exc).__name__+': '+str(exc)
        finally:
            with self._changed:
                task.ended=time.monotonic()
                if self._active_task is task:self._active=None;self._active_task=None
                self._changed.notify_all()
        # Never retain arrays in completed Future results or tracebacks.

    def _wait(self,key,cancelled=None,progress=None):
        previous=None
        while True:
            if cancelled and cancelled():self.cancel(key);_cancelled(lambda:True)
            result=self.get(key) if isinstance(key,PhotoKey) else self.get_source(key)
            if result is not None:return result
            value=self.status(key)
            if progress and value['stage']!=previous:progress(value['stage']);previous=value['stage']
            if value['state']=='cancelled':_cancelled(lambda:True)
            if value['state']=='failed':
                if value['error_code']=='source_changed':raise SourceChanged(value['error'])
                raise RuntimeError(value['error'])
            if value['state'] in ('completed','missing'):raise RuntimeError('工作副本已移出缓存，请重新请求')
            with self._changed:self._changed.wait(.04)

    def prepare(self,path,cancelled=None,progress=None):
        return self._wait(self.request_prepare(path),cancelled,progress)

    def get_or_compute(self,path,model='nafnet',backend_mode='auto',cancelled=None,progress=None,*,supersede=True):
        if supersede:return self._wait(self.request(path,model,backend_mode),cancelled,progress)
        key=key_for(path,model,backend_mode);leased=None
        # Request and lease registration are atomic with respect to a later
        # UI supersede. A cached Result already has its normal file/array lease.
        with self._changed:
            self._request(key,supersede=False)
            task=self._tasks.get(key)
            if self._get_cached(key) is None and task is not None:
                task.protected_consumers+=1;leased=task;self._changed.notify_all()
        try:return self._wait(key,cancelled,progress)
        finally:
            if leased is not None:
                with self._changed:
                    leased.protected_consumers-=1
                    # If this consumer failed/abandoned its wait, a deferred
                    # replacement may now stop unneeded inference. A published
                    # result remains valid, even during its tiny final handoff.
                    if not leased.protected_consumers and leased.supersede_deferred and self._get_cached(key) is None:
                        self._cancel_task(leased,'superseded')
                    self._changed.notify_all()

    def cancel(self,key=None):
        with self._changed:
            for task in self._history.values():
                if key is None or task.key==key:self._cancel_task(task)
            self._changed.notify_all()

    def cancel_task(self,task_id):
        """Cancel one visible queue row, including an older superseded task."""
        with self._changed:
            task=self._history.get(task_id)
            if task is None or task.state not in ('queued','preparing','running'):return False
            self._cancel_task(task);self._changed.notify_all()
            return True

    def cache_info(self):
        with self._lock:return {'sources':len(self._sources),'results':len(self._results),'retained_bytes':self._bytes(),
                                'max_bytes':self.max_bytes,'max_entries':self.max_entries,'active':self._active}

    def close(self,wait=True):
        with self._lock:self._closed=True
        self.cancel();self._pool.shutdown(wait=wait,cancel_futures=True)
        with self._changed:
            dropped=list(self._results.values())
            self._sources.clear();self._results.clear();self._previews.clear();self._changed.notify_all()
        # Result finalizers may remove files or briefly retry Windows handles.
        # Do not hold the status/cache lock across filesystem cleanup.
        dropped.clear()
