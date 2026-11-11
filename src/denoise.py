"""Optional original DCU module and local FP32 neural photographic denoising.

Device selection changes only execution backend. Model, original resolution,
precision, context and overlap remain fixed. One completed full image is cached.
"""
import hashlib,json,os,sys,threading,time
from contextlib import contextmanager
import numpy as np
from app_paths import ROOT

_lock=threading.RLock();_sessions={};_session_info={};_state={};_cache=None
_disabled_adapters=set()
DEFAULT_MODEL='dcu'
MODELS={
    'dcu':(None,None,None,None),
    'nafnet':('nafnet_sidd64_384.onnx','nafnet-reference-384.npz','nafnet-manifest-384.json',64),
    'restormer':('restormer_real_384.onnx','restormer-reference-384.npz','restormer-manifest-384.json',64),
}
# tile size, discarded context per edge, overlap of retained centres.
TILING={'nafnet':(384,64,64),'restormer':(384,64,64)}
LARGE_MODELS={}
LARGE_TILING={}
REFERENCE_TOLERANCE=1e-4


class InvalidDenoiseOutput(RuntimeError):
    pass


def _validate_prediction(result,shape):
    """Reject numerical failure before clipping; allow normal model overshoot.

    Inputs are normalized to [0, 1]. The broad [-1, 2] envelope permits a
    whole signal range of overshoot on either side. This is a numerical
    corruption guard, not a colour correction or a denoising strength rule.
    """
    if result.shape!=shape or not np.isfinite(result).all():
        raise InvalidDenoiseOutput('降噪输出无效：尺寸或数值异常')
    low=float(result.min());high=float(result.max())
    if low < -1. or high > 2.:
        # AUDIT-P38: state the actual limitation instead of a raw numeric range.
        # These models were trained on normally exposed sRGB frames; on photos whose
        # average lightness is low they can diverge, and that is not a corrupt file.
        raise InvalidDenoiseOutput(
            f'降噪输出数值异常（范围 {low:.4g} ~ {high:.4g}）。'
            'NAFNet / Restormer 只适用于明度较高的照片；'
            '这张照片暗部占比过大，请改用 DCU 原厂降噪，或先提高曝光再降噪。')


def runtime():
    bundled=ROOT/'neural-runtime'
    if bundled.is_dir() and str(bundled) not in sys.path:sys.path.insert(0,str(bundled))
    import onnxruntime as ort
    ort.disable_telemetry_events();ort.set_default_logger_severity(3)
    return ort


def status():return dict(_state)


def report_phase(phase,**values):
    _state.update(phase=phase,**values)


def _progress(callback,text,phase=None):
    if phase is not None:report_phase(phase)
    if callback:callback(text)


def cache_info():
    cached=_cache
    if cached is None:return {'ready':False}
    return {'ready':True,'model':cached[1],'shape':cached[4].shape,
            'dtype':cached[4].dtype.name,'bytes':cached[4].nbytes}


def clear_cache():
    global _cache
    with _lock:_cache=None


def _check_cancel(cancelled):
    if cancelled and cancelled():
        from native_backend import RenderCancelled
        raise RenderCancelled()


@contextmanager
def _claim(cancelled,progress):
    """Wait cancellably when a cancelled local preview still owns inference."""
    waiting=False
    while not _lock.acquire(timeout=.04):
        _check_cancel(cancelled)
        if not waiting and progress:progress('等待前一个降噪任务退出…')
        waiting=True
    try:
        _check_cancel(cancelled)
        yield
    finally:_lock.release()


def _mode(force_cpu=False):
    return 'cpu' if force_cpu or os.environ.get('PHOTO_GPU_DISABLE')=='1' else 'auto'


def _spec(model,variant='base'):
    return LARGE_MODELS[model] if variant=='large' else MODELS[model]


def _tile_config(model,variant='base'):
    return LARGE_TILING[model] if variant=='large' else TILING[model]


def tile_plan(shape,model='nafnet',variant='base'):
    tile,margin,overlap=_tile_config(model,variant);core=tile-2*margin
    h,w=shape[:2];rows=len(_positions(h,core,overlap));columns=len(_positions(w,core,overlap))
    return {'tile':tile,'context':margin,'overlap':overlap,'core':core,'stride':core-overlap,
            'rows':rows,'columns':columns,'tiles':rows*columns,'processed_pixels':rows*columns*tile*tile}


def _choose_variant(model,shape,mode,adapters):
    if model not in LARGE_MODELS or shape is None:return 'base'
    from neural_devices import memory_available,large_tile_allowed,blocked
    if not all((ROOT/'denoise-data'/f).is_file() for f in LARGE_MODELS[model][:3]):return 'base'
    a=tile_plan(shape,model);b=tile_plan(shape,model,'large')
    # Small detail windows retain the smaller profile. A large image must save
    # substantial actual model work, not just display fewer progress updates.
    if max(shape[:2])<512 or b['processed_pixels']>a['processed_pixels']*.8:return 'base'
    if memory_available()<3*1024**3:return 'base'
    if mode!='cpu' and not any(large_tile_allowed(v,384) and not blocked(v,model,'large') for v in adapters):return 'base'
    return 'large'


def _device_order(devices,mode):
    from neural_devices import CPU_KEY,priority_order
    if mode=='cpu':return [CPU_KEY]
    return priority_order(devices)


def _session_key(model,mode,variant,order):
    return model,mode,variant,tuple(order)


def _create(model,providers,variant='base'):
    ort=runtime();opts=ort.SessionOptions();opts.enable_mem_pattern=False
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads=min(8,os.cpu_count() or 1);opts.inter_op_num_threads=1;opts.log_severity_level=3
    is_dml=any((provider[0] if isinstance(provider,tuple) else provider)=='DmlExecutionProvider' for provider in providers)
    if model in ('restormer','nafnet'):
        # Split GPU dispatches to avoid driver limits of one large fused graph.
        # No original operation or FP32 weight is approximated or removed.
        opts.add_session_config_entry('ep.dml.disable_graph_fusion','1')
        opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    engine=ort.InferenceSession(str(ROOT/'denoise-data'/_spec(model,variant)[0]),opts,providers=providers)
    engine.disable_fallback()
    tile=_tile_config(model,variant)[0]
    if engine.get_inputs()[0].shape!=[1,3,tile,tile]:raise RuntimeError('降噪模型输入尺寸与分块设置不一致')
    return engine


def _run(engine,feed):
    # The interactive OpenCL colour pipeline shares the same physical GPU.
    # Serialize only GPU execution, not CPU processing or UI interaction.
    if engine.get_providers()[0]=='DmlExecutionProvider':
        import gpu_compute
        with gpu_compute._lock:return engine.run(None,feed)[0]
    return engine.run(None,feed)[0]


def session(model=DEFAULT_MODEL,force_cpu=False,cancelled=None,progress=None,shape=None,variant=None,
            skip_devices=()):
    if model not in MODELS:raise ValueError('不支持的降噪模型')
    mode='cpu' if model=='dcu' else _mode(force_cpu)
    with _lock:
        _check_cancel(cancelled)
        from neural_devices import CPU_KEY,adapters as list_adapters,blocked,record_failure,large_tile_allowed,write_diagnostic,failure
        devices=[]
        if model!='dcu' and mode!='cpu':
            try:devices=list_adapters()
            except (OSError,RuntimeError):devices=[]
        order=_device_order(devices,mode) if model!='dcu' else [CPU_KEY]
        rows={row['key']:row for row in devices}
        selection_mode='auto' if order and order[0]!=CPU_KEY else 'cpu'
        variant=variant or (_choose_variant(model,shape,selection_mode,devices) if model!='dcu' else 'base')
        key=_session_key(model,mode,variant,order)
        # Model switching must not retain all large CPU/GPU workspaces.
        expired=[k for k in _sessions if len(k)!=4 or k[0]!=model or k[1]!=mode or
                 k[2]!=variant or k[3]!=tuple(order)]
        if expired and progress:progress('正在释放上一个模型…')
        for old_key in expired:
            _sessions.pop(old_key,None);_session_info.pop(old_key,None)
        if model=='dcu':
            _state.clear();_state.update(model='dcu',backend='CPU · DCU',native_module_defaults=True)
            return None
        if key in _sessions:
            cached_info=_session_info[key];adapter=cached_info.get('adapter')
            if adapter and (adapter['key'] in skip_devices or adapter['key'] in _disabled_adapters
                            or blocked(adapter,model,variant)):
                _sessions.pop(key,None);_session_info.pop(key,None)
            else:
                _state.clear();_state.update(cached_info);return _sessions[key]
        filename,fixture,manifest,_=_spec(model,variant);root=ROOT/'denoise-data'
        name={'nafnet':'NAFNet','restormer':'Restormer'}[model]
        _state.clear();_state.update(model=model,backend='准备中',variant=variant,provider=None)
        _progress(progress,name+' · 校验模型文件…','checking_model')
        meta=json.loads((root/manifest).read_text(encoding='utf-8'))
        with (root/filename).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=meta['onnx_sha256']:raise ValueError('降噪模型校验失败，请重新下载完整程序')
        _check_cancel(cancelled)
        with np.load(root/fixture,allow_pickle=False) as f:feed={'image':f['image']};reference=f['clean']
        _progress(progress,name+' · 准备运行环境…','loading_runtime')
        ort=runtime();_check_cancel(cancelled)
        tested=[]
        common={'model':model,'variant':variant,'tile_config':list(_tile_config(model,variant)),
                'model_sha256':digest,'precision':'float32','phase':'validating_device','eta_seconds':None}
        selected=None
        dml_available='DmlExecutionProvider' in ort.get_available_providers()
        for device_key in order:
            _check_cancel(cancelled)
            if device_key==CPU_KEY:
                _state.clear();_state.update(model=model,variant=variant,backend='CPU',provider='CPUExecutionProvider')
                _progress(progress,name+' · 加载 CPU 模型…','loading_model')
                try:cpu=_create(model,['CPUExecutionProvider'],variant)
                except Exception as exc:
                    if variant=='large' and failure(exc)['kind']=='memory':
                        if progress:progress('较大分块内存不足，使用同一模型的标准分块…')
                        return session(model,force_cpu,cancelled,progress,shape,'base',skip_devices)
                    raise
                _progress(progress,name+' · 校验 CPU 模型…','validating_device')
                result=_run(cpu,feed);_check_cancel(cancelled)
                error=float(np.max(np.abs(result-reference))) if result.shape==reference.shape else float('inf')
                valid=bool(result.shape==reference.shape and np.isfinite(result).all() and error<=REFERENCE_TOLERANCE)
                tested.append({'device':'CPU','provider':'CPUExecutionProvider','max_error':error,'valid':valid})
                if not valid:raise RuntimeError('CPU 降噪模型自检失败')
                selected=(cpu,'CPU',None,'CPUExecutionProvider')
                break
            info=rows.get(device_key)
            if info is None:continue
            if not dml_available:
                tested.append({'device':info['name'],'device_id':info['id'],'available':False,
                               'reason':'当前运行环境没有 DirectML 执行提供程序。'});continue
            unavailable=blocked(info,model,variant)
            if device_key in _disabled_adapters or device_key in skip_devices:
                unavailable={'kind':'device','reason':'保留此前设备故障保护，未自动解禁。'}
            if variant=='large' and not large_tile_allowed(info,384):
                unavailable={'kind':'memory_budget','reason':'当前显存预算不足以安全使用较大分块。'}
            if unavailable:
                tested.append({'device':info['name'],'device_id':info['id'],'available':False,**unavailable});continue
            options={'device_id':str(info['id']),'disable_metacommands':'True'}
            _state.update(provider='DmlExecutionProvider',adapter=dict(info),
                          backend=info['name'],model=model,variant=variant)
            _progress(progress,name+' · 加载 GPU 模型 · '+info['name'],'loading_model')
            try:
                gpu=_create(model,[('DmlExecutionProvider',options),'CPUExecutionProvider'],variant)
                if gpu.get_providers()[0]!='DmlExecutionProvider':
                    tested.append({'device':info['name'],'device_id':info['id'],'available':False,
                                   'reason':'DirectML 未成功绑定该设备。'});continue
                _progress(progress,name+' · 校验 GPU 模型 · '+info['name'],'validating_device')
                result=_run(gpu,feed);_check_cancel(cancelled)
                error=float(np.max(np.abs(result-reference))) if result.shape==reference.shape else float('inf')
                valid=bool(result.shape==reference.shape and np.isfinite(result).all() and error<=REFERENCE_TOLERANCE)
                tested.append({'device':info['name'],'device_id':info['id'],'provider':'DmlExecutionProvider',
                               'max_error':error,'valid':valid,
                               'reason':'模型校验通过，按设备优先级选用。' if valid else '模型输出未通过精度校验。'})
                if valid:
                    selected=(gpu,info['name'],info,'DmlExecutionProvider')
                    break
            except Exception as exc:
                from native_backend import RenderCancelled
                if isinstance(exc,RenderCancelled):raise
                problem=record_failure(info,model,variant,exc)
                tested.append({'device':info['name'],'device_id':info['id'],**problem})
                if problem['kind']=='device':_disabled_adapters.add(device_key)
                if variant=='large' and problem['kind']=='memory':
                    if progress:progress('较大分块内存不足，使用同一模型的标准分块…')
                    return session(model,force_cpu,cancelled,progress,shape,'base',skip_devices)
                continue
        if selected is None:raise RuntimeError('没有可用的 AI 降噪设备；请检查设备顺序和模型运行环境。')
        _check_cancel(cancelled)
        engine,backend,adapter,provider=selected
        info=dict(common,tested=tested,backend=backend,adapter=adapter,provider=provider,
                  selection_reason='按设置中的设备顺序选择首个通过模型精度校验的设备。')
        _sessions[key]=engine;_session_info[key]=info
        _state.clear();_state.update(info)
        write_diagnostic(info)
        return engine


def _reflect_indices(start,count,length):
    if length==1:return np.zeros(count,dtype=np.intp)
    indices=np.arange(start,start+count,dtype=np.int64)%(2*(length-1))
    return np.where(indices<length,indices,2*(length-1)-indices)


def _positions(length,core,overlap):
    if length<=core:return [0]
    step=core-overlap;positions=list(range(0,length-core+1,step))
    # Preserve the encoder/downsampling phase at the last position.
    last=((length-core+7)//8)*8
    if last>positions[-1]:positions.append(last)
    return positions


def _content_key(rgb,cancelled=None):
    digest=hashlib.blake2b(digest_size=16)
    for top in range(0,len(rgb),128):
        _check_cancel(cancelled);digest.update(np.ascontiguousarray(rgb[top:top+128]))
    return (rgb.shape,rgb.dtype.str,digest.digest())


def _infer(rgb,model,noise_level,cancelled,progress,force_cpu=False,variant=None,skip_devices=()):
    _check_cancel(cancelled)
    engine=session(model,force_cpu,cancelled,progress,shape=rgb.shape,variant=variant,
                   skip_devices=skip_devices)
    if model=='dcu':
        from dcu_denoise import apply as dcu_apply
        return dcu_apply(rgb,cancelled,progress)
    if progress:progress('正在准备图像分块 · '+_state.get('backend','CPU'))
    maximum=np.iinfo(rgb.dtype).max;tile,margin,overlap=_state.get('tile_config',TILING[model])
    core=tile-2*margin;h,w=rgb.shape[:2];tops=_positions(h,core,overlap);lefts=_positions(w,core,overlap)
    out=np.empty_like(rgb)
    # One row band of FP32 accumulators, rather than a full-image FP32 copy.
    # Smooth overlap avoids hard cut boundaries without reducing resolution.
    band=np.zeros((core,w,3),np.float32);weight=np.zeros((core,w),np.float32)
    ramp=np.ones(core,np.float32)
    fade=np.sin(np.linspace(0,np.pi/2,overlap+2,dtype=np.float32)[1:-1])**2
    ramp[:overlap]=fade;ramp[-overlap:]=fade[::-1];window=ramp[:,None]*ramp[None,:]
    monochrome=np.array_equal(rgb[...,0],rgb[...,1]) and np.array_equal(rgb[...,0],rgb[...,2])
    total=len(tops)*len(lefts);done=0;inference_started=time.monotonic()
    _state.update(tiles_total=total,tiles_done=0,tile=tile,context=margin,overlap=overlap,
                  source_width=w,source_height=h,seconds_remaining=None,phase='processing_tiles',
                  done=0,total=total,elapsed_seconds=0.,eta_seconds=None)
    if progress:progress('AI 降噪 0% · '+_state.get('backend','CPU'))
    for row_number,top in enumerate(tops):
        yi=_reflect_indices(top-margin,tile,h)
        for left in lefts:
            _check_cancel(cancelled);xi=_reflect_indices(left-margin,tile,w)
            patch=rgb[yi[:,None],xi[None,:]]
            a=np.ascontiguousarray(patch.transpose(2,0,1)[None],dtype=np.float32)/maximum
            result=_run(engine,{'image':a})
            _validate_prediction(result,(1,3,tile,tile))
            hh=min(core,h-top);ww=min(core,w-left)
            prediction=result[0,:,margin:margin+hh,margin:margin+ww].transpose(1,2,0);local_weight=window[:hh,:ww]
            band[:hh,left:left+ww]+=prediction*local_weight[...,None];weight[:hh,left:left+ww]+=local_weight
            done+=1
            seconds_remaining=(time.monotonic()-inference_started)/done*(total-done)
            _state.update(tiles_done=done,seconds_remaining=round(seconds_remaining,1),done=done,
                          elapsed_seconds=round(time.monotonic()-inference_started,1),eta_seconds=round(seconds_remaining,1))
            if progress:
                eta=(f'预计剩余 {max(1,round(seconds_remaining/60))} 分钟' if seconds_remaining>=60 else f'预计剩余 {max(1,round(seconds_remaining))} 秒')
                progress(f'AI 降噪 {min(99,done*100//total)}% · {eta} · '+_state.get('backend','CPU'))
        flush=(tops[row_number+1]-top) if row_number+1<len(tops) else h-top
        if np.any(weight[:flush]<=0):raise RuntimeError('降噪分块覆盖不完整')
        values=band[:flush]/weight[:flush,:,None]
        if monochrome:
            luminance=values[...,0].astype(np.float64)*.2126+values[...,1].astype(np.float64)*.7152+values[...,2].astype(np.float64)*.0722
            values=np.repeat(luminance[...,None],3,axis=-1)
        out[top:top+flush]=np.floor(np.clip(values.astype(np.float64),0,1)*maximum+.5).astype(rgb.dtype)
        if row_number+1<len(tops):
            remain=core-flush
            if remain:
                band[:remain]=band[flush:].copy();weight[:remain]=weight[flush:].copy()
            band[remain:]=0;weight[remain:]=0
    _check_cancel(cancelled)
    return out


def _infer_with_recovery(rgb,model,noise_level,cancelled,progress,force_cpu,variant,skip_devices):
    """Finish on the first usable device; each failed device is tried once.

    Session creation already follows the saved order. Runtime failure uses
    that same order, preserving the model, FP32 precision and pixel input.
    CPU failure is terminal. A large-tile memory failure may retry base tiles
    once before advancing to another device.
    """
    from native_backend import RenderCancelled
    from neural_devices import failure,record_failure,write_diagnostic,adapter_key
    skipped=set(skip_devices);fallbacks=[]
    while True:
        try:
            clean=_infer(rgb,model,noise_level,cancelled,progress,force_cpu=force_cpu,
                         variant=variant,skip_devices=skipped)
            if fallbacks:
                _state.update(fallback_reason=fallbacks[-1]['reason'],
                              fallback_kind=fallbacks[-1]['kind'],fallbacks=fallbacks)
                write_diagnostic(_state)
            return clean
        except RenderCancelled:raise
        except Exception as exc:
            info=dict(_state);adapter=info.get('adapter')
            current_variant=info.get('variant','base')
            problem=failure(exc)
            if adapter:problem=record_failure(adapter,model,current_variant,exc)
            retry_small=current_variant=='large' and problem['kind']=='memory'
            if not retry_small and info.get('provider')!='DmlExecutionProvider':
                write_diagnostic(dict(info,phase='failed',failure=problem))
                raise
            device_key=(adapter.get('key') or adapter_key(adapter)) if adapter else None
            # Every loop either drops the large profile, skips a new device,
            # or forces CPU. A misreported backend cannot cause an endless retry.
            if not retry_small and device_key in skipped:raise
            if not retry_small and device_key is None and force_cpu:raise
            if device_key and problem['kind']=='device':_disabled_adapters.add(device_key)
            if not retry_small and device_key:skipped.add(device_key)
            for key in list(_sessions):
                if key[0]==model:
                    _sessions.pop(key,None);_session_info.pop(key,None)
            exc.__traceback__=None
            fallbacks.append(dict(problem,backend=info.get('backend','CPU')))
            if progress:progress('较大分块内存不足，改用同一模型标准分块…' if retry_small
                                 else '当前设备未能完成降噪，正在按设备顺序重试…')
            variant='base' if retry_small else current_variant
            if not retry_small and not device_key:force_cpu=True


def apply(rgb,amount=1,cancelled=None,progress=None,model=DEFAULT_MODEL,noise_level=15,
          variant=None,force_cpu=False,skip_devices=()):
    """Return original-size RGB8/16, with a cached read-only 100% result.

    Input is display sRGB. noise_level remains for old settings compatibility;
    these blind models have no manually invented noise-level input. amount only
    blends the completed output; no hardware or strength setting changes quality.
    """
    global _cache
    from native_backend import RenderCancelled
    if model not in MODELS or not np.isfinite(noise_level) or not 1<=noise_level<=50:raise ValueError('降噪设置无效')
    if not np.isfinite(amount) or not 0<=amount<=1:raise ValueError('降噪强度无效')
    if not amount:return rgb
    if rgb.dtype not in (np.uint8,np.uint16) or rgb.ndim!=3 or rgb.shape[-1]!=3 or not all(rgb.shape[:2]):raise ValueError('降噪需要非空 RGB8 / RGB16')
    _check_cancel(cancelled)
    _progress(progress,'正在准备降噪输入…','preparing_input')
    with _claim(cancelled,progress):
        _check_cancel(cancelled);mode='cpu' if model=='dcu' else _mode(force_cpu);key=_content_key(rgb,cancelled);cached=_cache
        if cached is not None and cached[:4]==(key,model,0,mode):clean=cached[4]
        else:
            remote=model!='dcu' and os.environ.get('PHOTO_AI_WORKER')!='1'
            if remote:
                from neural_process import infer
                def receive_status(info):
                    _state.clear();_state.update(info)
                clean,info=infer(rgb,model,mode,cancelled,progress,on_status=receive_status);_state.clear();_state.update(info)
            else:clean=_infer_with_recovery(rgb,model,noise_level,cancelled,progress,
                                           force_cpu,variant,skip_devices)
            _check_cancel(cancelled);clean.flags.writeable=False;_cache=(key,model,0,mode,clean)
        _check_cancel(cancelled)
        if amount==1:
            report_phase('completed',eta_seconds=0.)
            if progress:progress(('DCU' if model=='dcu' else 'AI')+' 降噪 100% · 完成')
            return clean
        out=np.empty_like(rgb);maximum=np.iinfo(rgb.dtype).max
        for row in range(0,len(rgb),128):
            _check_cancel(cancelled);a=rgb[row:row+128].astype(np.float64);b=clean[row:row+128].astype(np.float64)
            out[row:row+128]=np.floor(np.clip(a+(b-a)*amount,0,maximum)+.5).astype(rgb.dtype)
        if progress:progress(('DCU' if model=='dcu' else 'AI')+' 降噪 100% · 完成')
        report_phase('completed',eta_seconds=0.)
        return out
