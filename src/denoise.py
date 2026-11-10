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
_gpu_runtime_disabled=False
_variant_timings={}
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
    import gpu_compute
    return 'cpu' if force_cpu or os.environ.get('PHOTO_GPU_DISABLE')=='1' or gpu_compute.status()['mode']=='cpu' else 'auto'


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
    base_time=_variant_timings.get((model,mode,'base'));large_time=_variant_timings.get((model,mode,'large'))
    if base_time and large_time and large_time*b['tiles']>=base_time*a['tiles']*.95:return 'base'
    return 'large'


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


def session(model=DEFAULT_MODEL,force_cpu=False,cancelled=None,progress=None,shape=None,variant=None):
    global _gpu_runtime_disabled
    if model not in MODELS:raise ValueError('不支持的降噪模型')
    mode='cpu' if model=='dcu' else _mode(force_cpu)
    with _lock:
        _check_cancel(cancelled)
        from neural_devices import adapters as list_adapters,blocked,record_failure,large_tile_allowed,write_diagnostic,failure
        devices=[]
        if model!='dcu' and mode!='cpu' and not _gpu_runtime_disabled:
            try:devices=list_adapters()
            except OSError:devices=[]
        variant=variant or (_choose_variant(model,shape,mode,devices) if model!='dcu' else 'base')
        key=(model,mode) if variant=='base' else (model,mode,variant)
        cpu_key=(model,'cpu') if variant=='base' else (model,'cpu',variant)
        # Model switching must not retain all large CPU/GPU workspaces.
        expired=[k for k in _sessions if k[0]!=model or (k[2] if len(k)>2 else 'base')!=variant]
        if expired and progress:progress('正在释放上一个模型…')
        for old_key in expired:
            _sessions.pop(old_key,None);_session_info.pop(old_key,None)
        if model=='dcu':
            _state.clear();_state.update(model='dcu',backend='CPU · DCU',native_module_defaults=True)
            return None
        if key in _sessions:
            _state.clear();_state.update(_session_info[key]);return _sessions[key]
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
        runtime();_check_cancel(cancelled)
        _progress(progress,name+' · 加载 CPU 模型…','loading_model')
        try:cpu=_create(model,['CPUExecutionProvider'],variant)
        except Exception as exc:
            if variant=='large' and failure(exc)['kind']=='memory':
                if progress:progress('较大分块内存不足，使用同一模型的标准分块…')
                return session(model,force_cpu,cancelled,progress,shape,'base')
            raise
        _check_cancel(cancelled)
        _progress(progress,name+' · 预热 CPU 模型…','validating_device')
        _run(cpu,feed);_check_cancel(cancelled)
        if progress:progress(name+' · 校验 CPU 精度…')
        samples=[]
        for _ in range(2):
            _check_cancel(cancelled);start=time.perf_counter();result=_run(cpu,feed);samples.append(time.perf_counter()-start)
        cpu_time=float(np.median(samples))
        error=float(np.max(np.abs(result-reference)))
        if not np.isfinite(result).all() or error>REFERENCE_TOLERANCE:raise RuntimeError('CPU 降噪模型自检失败')
        tested=[{'device':'CPU','seconds':cpu_time,'timings':samples,'max_error':error,'valid':True}]
        common={'model':model,'variant':variant,'tile_config':list(_tile_config(model,variant)),'tested':tested,
                'model_sha256':digest,'precision':'float32','phase':'validating_device','eta_seconds':None}
        cpu_info=dict(common,backend='CPU',provider='CPUExecutionProvider',seconds_per_tile=cpu_time)
        _state.clear();_state.update(cpu_info)
        if _gpu_runtime_disabled:tested.append({'device':'GPU','available':False,'kind':'device','reason':'保留本进程此前设备故障保护，未自动解禁。'})
        best=(cpu_time,cpu,'CPU',None)
        if mode!='cpu' and not _gpu_runtime_disabled and 'DmlExecutionProvider' in runtime().get_available_providers():
            if progress:progress(name+' · 检测可用显卡…')
            seen=set()
            for info in devices:
                _check_cancel(cancelled)
                ident=(info['name'],info['vendor'],info['dedicated'])
                physical=info.get('luid',ident)
                if physical in seen:continue
                seen.add(physical)
                unavailable=blocked(info,model,variant)
                if ident in _disabled_adapters:unavailable={'kind':'device','reason':'保留此前设备故障保护，未自动解禁。'}
                if variant=='large' and not large_tile_allowed(info,384):unavailable={'kind':'memory_budget','reason':'当前显存预算不足以安全使用较大分块。'}
                if unavailable:
                    tested.append({'device':info['name'],'device_id':info['id'],'available':False,**unavailable});continue
                # Portable FP32 kernels and split dispatches avoid the known
                # fused/metacommand driver fault; do not retry that unsafe path.
                for portable in (True,):
                    try:
                        options={'device_id':str(info['id'])}
                        if portable:options['disable_metacommands']='True'
                        # Publish the exact risky configuration before entering
                        # native code. An access violation cannot be caught here;
                        # the parent needs this context to quarantine the profile.
                        _state.update(provider='DmlExecutionProvider',adapter=dict(info),
                                      backend=info['name'],model=model,variant=variant)
                        _progress(progress,name+' · 加载 GPU 模型 · '+info['name'],'loading_model')
                        gpu=_create(model,[('DmlExecutionProvider',options),'CPUExecutionProvider'],variant)
                        if gpu.get_providers()[0]!='DmlExecutionProvider':
                            tested.append({'device':info['name'],'device_id':info['id'],'available':False,'reason':'DirectML 未成功绑定该设备。'});break
                        _check_cancel(cancelled)
                        _progress(progress,name+' · 预热 GPU 模型 · '+info['name'],'validating_device')
                        _run(gpu,feed);_check_cancel(cancelled)
                        if progress:progress(name+' · 校验 GPU 精度 · '+info['name'])
                        samples=[]
                        for _ in range(2):
                            _check_cancel(cancelled);start=time.perf_counter();result=_run(gpu,feed);samples.append(time.perf_counter()-start)
                        elapsed=float(np.median(samples))
                        error=float(np.max(np.abs(result-reference)));valid=bool(np.isfinite(result).all() and error<=REFERENCE_TOLERANCE)
                        tested.append({'device':info['name'],'device_id':info['id'],'seconds':elapsed,'timings':samples,'max_error':error,'valid':valid,'portable_kernels':portable,
                                       'reason':'精度通过，按当前同模型实测速度比较。' if valid else '结果未通过原 PyTorch 精度门槛。'})
                        if valid:
                            if elapsed<best[0]*.95:best=(elapsed,gpu,info['name'],info)
                            break
                    except Exception as exc:
                        from native_backend import RenderCancelled
                        if isinstance(exc,RenderCancelled):raise
                        problem=record_failure(info,model,variant,exc)
                        tested.append({'device':info['name'],'device_id':info['id'],'available':False,'portable_kernels':portable,**problem})
                        # A driver/device exception is different from a finite
                        # numeric mismatch: do not immediately retry a hung GPU.
                        if problem['kind']=='device':
                            _disabled_adapters.add(ident);_gpu_runtime_disabled=True
                            best=(cpu_time,cpu,'CPU',None)
                        break
                if _gpu_runtime_disabled:break
        _check_cancel(cancelled)
        info=dict(common,backend=best[2],seconds_per_tile=best[0],adapter=best[3],
                  provider='CPUExecutionProvider' if best[3] is None else 'DmlExecutionProvider',
                  selection_reason='使用通过同模型 FP32 精度校验的最快实测设备；故障或预算不足的配置保留跳过原因。')
        _variant_timings[(model,mode,variant)]=best[0]
        _sessions[key]=best[1];_session_info[key]=info
        _sessions.setdefault(cpu_key,cpu);_session_info.setdefault(cpu_key,cpu_info)
        if variant=='large' and shape is not None:
            baseline_time=_variant_timings.get((model,mode,'base'))
            if baseline_time and best[0]*tile_plan(shape,model,'large')['tiles']>=baseline_time*tile_plan(shape,model)['tiles']*.95:
                # Compare expected whole-image work with the already measured
                # smaller profile, not isolated tile counts. Release large
                # allocations before restoring the faster standard profile.
                for old_key in [k for k in _sessions if len(k)>2 and k[2]=='large']:
                    _sessions.pop(old_key,None);_session_info.pop(old_key,None)
                cpu=None;gpu=None;best=None;result=None;feed=None;reference=None
                _progress(progress,'较大分块实测没有加快，使用同一模型的标准分块…','selecting_device')
                engine=session(model,force_cpu,cancelled,progress,shape,'base')
                _state['tile_decision_reason']='较大分块的实测预计总耗时更长，已选择标准分块。'
                write_diagnostic(_state)
                return engine
        _state.clear();_state.update(info)
        write_diagnostic(info)
        return best[1]


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


def _infer(rgb,model,noise_level,cancelled,progress,force_cpu=False,variant=None):
    _check_cancel(cancelled)
    engine=session(model,force_cpu,cancelled,progress,shape=rgb.shape,variant=variant)
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
            if result.shape!=(1,3,tile,tile) or not np.isfinite(result).all():raise RuntimeError('降噪输出无效')
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


def apply(rgb,amount=1,cancelled=None,progress=None,model=DEFAULT_MODEL,noise_level=15,variant=None):
    """Return original-size RGB8/16, with a cached read-only 100% result.

    Input is display sRGB. noise_level remains for old settings compatibility;
    these blind models have no manually invented noise-level input. amount only
    blends the completed output; no hardware or strength setting changes quality.
    """
    global _cache,_gpu_runtime_disabled
    from native_backend import RenderCancelled
    if model not in MODELS or not np.isfinite(noise_level) or not 1<=noise_level<=50:raise ValueError('降噪设置无效')
    if not np.isfinite(amount) or not 0<=amount<=1:raise ValueError('降噪强度无效')
    if not amount:return rgb
    if rgb.dtype not in (np.uint8,np.uint16) or rgb.ndim!=3 or rgb.shape[-1]!=3 or not all(rgb.shape[:2]):raise ValueError('降噪需要非空 RGB8 / RGB16')
    _check_cancel(cancelled)
    _progress(progress,'正在准备降噪输入…','preparing_input')
    with _claim(cancelled,progress):
        _check_cancel(cancelled);mode='cpu' if model=='dcu' else _mode();key=_content_key(rgb,cancelled);cached=_cache
        if cached is not None and cached[:4]==(key,model,0,mode):clean=cached[4]
        else:
            remote=model!='dcu' and os.environ.get('PHOTO_AI_WORKER')!='1'
            try:
                if remote:
                    from neural_process import infer
                    def receive_status(info):
                        _state.clear();_state.update(info)
                    clean,info=infer(rgb,model,mode,cancelled,progress,on_status=receive_status);_state.clear();_state.update(info)
                else:clean=_infer(rgb,model,noise_level,cancelled,progress,variant=variant)
            except RenderCancelled:raise
            except Exception as exc:
                # The isolated worker already performs the same-model CPU
                # retry. Do not create a second model session inside the GUI.
                if remote:raise
                from neural_devices import failure,record_failure,write_diagnostic
                problem=failure(exc);variant=_state.get('variant','base');adapter=_state.get('adapter')
                if adapter:problem=record_failure(adapter,model,variant,exc)
                if problem['kind']=='device':_gpu_runtime_disabled=True
                retry_small=variant=='large' and problem['kind']=='memory'
                if not retry_small and _state.get('provider')!='DmlExecutionProvider':raise
                # Invalidate auto sessions before a whole-image CPU restart;
                # subsequent photos must not revisit the same faulty adapter.
                for failed_key in [k for k in _sessions if k[1]=='auto' or retry_small]:
                    _sessions.pop(failed_key,None);_session_info.pop(failed_key,None)
                # Release a failed inference frame before allocating its
                # replacement; traceback locals can own GPU workspaces.
                exc.__traceback__=None
                if progress:progress('较大分块内存不足，改用同一模型标准分块…' if retry_small else 'GPU 配置不可用，正在准备同一模型的 CPU 处理…')
                clean=_infer(rgb,model,noise_level,cancelled,progress,force_cpu=not retry_small,variant='base' if retry_small else variant)
                _state.update(fallback_reason=problem['reason'],fallback_kind=problem['kind'])
                selected_variant=_state.get('variant','base')
                auto_key=(model,'auto') if selected_variant=='base' else (model,'auto',selected_variant)
                selected_key=(model,'cpu') if selected_variant=='base' else (model,'cpu',selected_variant)
                if not retry_small:
                    _sessions[auto_key]=_sessions[selected_key]
                    _session_info[auto_key]=dict(_state)
                write_diagnostic(_state)
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
