"""Optional cross-vendor OpenCL 1.2 FP64 kernels; checked CPU fallback.

Only the OS-installed ICD loader is used. No CUDA runtime, device identifiers,
driver installation, float32 approximation, or native DCU replacement.
"""
import ctypes as C
import hashlib,json,os,threading,time
from pathlib import Path
import numpy as np
from app_paths import ROOT

P=C.c_void_p;U=C.c_uint;S=C.c_size_t;Q=C.c_ulonglong;I=C.c_int
_lock=threading.RLock();_backend=None;_attempted=False;_mode='auto';_external_busy=False
_state={'state':'waiting','device':'','reason':'尚未检测','devices':[]}


def set_mode(mode):
    global _mode
    if mode not in ('auto','cpu'):raise ValueError('Unknown acceleration preference')
    _mode=mode


def status():
    return dict(_state,mode=_mode,external_busy=_external_busy)

def set_external_busy(busy):
    """Reserve the GPU for an isolated AI job; colour keeps its CPU reference.

    Called only on background workers. Existing OpenCL dispatch finishes before
    the reservation starts, with no user mode changes or cross-process races.
    """
    global _external_busy
    with _lock:_external_busy=bool(busy)


def _check(code):
    if code:raise RuntimeError(f'OpenCL error {code}')


class OpenCL:
    def __init__(self):
        if os.name!='nt':raise OSError('Windows OpenCL loader required')
        self.dll=C.WinDLL('OpenCL.dll',winmode=0x800) # LOAD_LIBRARY_SEARCH_SYSTEM32
        specs={
            'clGetPlatformIDs':(I,[U,C.POINTER(P),C.POINTER(U)]),
            'clGetDeviceIDs':(I,[P,Q,U,C.POINTER(P),C.POINTER(U)]),
            'clGetDeviceInfo':(I,[P,U,S,P,C.POINTER(S)]),
            'clCreateContext':(P,[P,U,C.POINTER(P),P,P,C.POINTER(I)]),
            'clCreateCommandQueue':(P,[P,P,Q,C.POINTER(I)]),
            'clCreateProgramWithSource':(P,[P,U,C.POINTER(C.c_char_p),C.POINTER(S),C.POINTER(I)]),
            'clBuildProgram':(I,[P,U,C.POINTER(P),C.c_char_p,P,P]),
            'clGetProgramBuildInfo':(I,[P,P,U,S,P,C.POINTER(S)]),
            'clCreateKernel':(P,[P,C.c_char_p,C.POINTER(I)]),
            'clCreateBuffer':(P,[P,Q,S,P,C.POINTER(I)]),
            'clSetKernelArg':(I,[P,U,S,P]),
            'clEnqueueWriteBuffer':(I,[P,P,U,S,S,P,U,P,P]),
            'clEnqueueReadBuffer':(I,[P,P,U,S,S,P,U,P,P]),
            'clEnqueueNDRangeKernel':(I,[P,P,U,P,C.POINTER(S),P,U,P,P]),
            'clFinish':(I,[P]),
        }
        for kind in ('Context','CommandQueue','Program','Kernel','MemObject'):
            specs['clRelease'+kind]=(I,[P])
        for name,(restype,argtypes) in specs.items():
            fn=getattr(self.dll,name);fn.restype=restype;fn.argtypes=argtypes;setattr(self,name,fn)

    def devices(self):
        n=U();_check(self.clGetPlatformIDs(0,None,C.byref(n)))
        ps=(P*n.value)();_check(self.clGetPlatformIDs(n,ps,None));rows=[]
        for platform in ps:
            count=U();code=self.clGetDeviceIDs(platform,4,0,None,C.byref(count))
            if code==-1:continue # no GPU on this platform
            _check(code);ds=(P*count.value)();_check(self.clGetDeviceIDs(platform,4,count,ds,None))
            for device in ds:
                def info(key,number=False):
                    buf=C.create_string_buffer(4096);length=S()
                    _check(self.clGetDeviceInfo(device,key,len(buf),buf,C.byref(length)))
                    return int.from_bytes(buf.raw[:length.value],'little') if number else buf.value.decode('utf-8','replace')
                rows.append(dict(handle=device,name=info(0x102b),vendor=info(0x102c),driver=info(0x102d),
                                 fp64=info(0x1032,True),max_alloc=info(0x1010,True),units=info(0x1002,True)))
        return rows


class Device:
    def __init__(self,api,info):
        self.api=api;self.info=info;self.lock=threading.RLock();self.owned=[];self.configs={}
        self.context=self.queue=self.program=None;self.kernels={};self.capacity=0;self.buffers=[]
        try:
            code=I();device=P(info['handle'])
            self.context=api.clCreateContext(None,1,C.byref(device),None,None,C.byref(code));_check(code.value)
            self.owned.append(('Context',self.context))
            self.queue=api.clCreateCommandQueue(self.context,device,0,C.byref(code));_check(code.value)
            self.owned.append(('CommandQueue',self.queue))
            source=(ROOT/'gr4-core/photo_kernels.cl').read_bytes()
            manifest=json.loads((ROOT/'gr4-data/manifest.json').read_text(encoding='utf-8'))
            if hashlib.sha256(source).hexdigest()!=manifest['opencl_source_sha256']:raise ValueError('GPU component checksum failed')
            src=C.c_char_p(source);length=S(len(source))
            self.program=api.clCreateProgramWithSource(self.context,1,C.byref(src),C.byref(length),C.byref(code));_check(code.value)
            self.owned.append(('Program',self.program))
            result=api.clBuildProgram(self.program,1,C.byref(device),b'-cl-std=CL1.2',None,None)
            if result:
                log=C.create_string_buffer(16384);api.clGetProgramBuildInfo(self.program,device,0x1183,len(log),log,None)
                raise RuntimeError(f'OpenCL build {result}: '+log.value.decode('utf-8','replace')[:1500])
            for name in ('gr4_u16','balance_u16','balance_u8'):
                kernel=api.clCreateKernel(self.program,name.encode(),C.byref(code));_check(code.value)
                self.kernels[name]=kernel;self.owned.append(('Kernel',kernel))
            # Bounded work buffers also avoid long single dispatches / VRAM spikes.
            self.chunk=min(1048576,max(4096,info['max_alloc']//48))
        except BaseException:self.close();raise

    def close(self):
        with self.lock:
            owned,self.owned=self.owned,[]
            for kind,handle in reversed(owned):
                # A failed device must not retain buffers, nor let a second
                # release failure prevent the remaining handles being freed.
                try:
                    if handle:getattr(self.api,'clRelease'+kind)(handle)
                except Exception:pass
            self.configs={};self.buffers=[];self.capacity=0

    def buffer(self,size,data=None):
        code=I();handle=self.api.clCreateBuffer(self.context,1,S(size),None,C.byref(code));_check(code.value)
        self.owned.append(('MemObject',handle))
        if data is not None:self.write(handle,size,data)
        return handle

    def write(self,handle,size,data):
        _check(self.api.clEnqueueWriteBuffer(self.queue,handle,1,0,size,data,0,None,None))

    def arg(self,kernel,index,value):
        _check(self.api.clSetKernelArg(kernel,index,C.sizeof(value),C.byref(value)))

    def execute(self,source,name,args,cancelled=None,progress=None):
        from native_backend import RenderCancelled
        source=np.ascontiguousarray(source);out=np.empty_like(source);total=source.size//3
        with self.lock:
            count=min(self.chunk,total);required=count*3*source.itemsize
            if required>self.capacity:
                for handle in self.buffers:
                    _check(self.api.clReleaseMemObject(handle));self.owned.remove(('MemObject',handle))
                self.buffers=[self.buffer(required),self.buffer(required)];self.capacity=required
            kernel=self.kernels[name]
            self.arg(kernel,0,P(self.buffers[0]));self.arg(kernel,1,P(self.buffers[1]))
            for i,arg in enumerate(args,3):self.arg(kernel,i,arg)
            stride=source.itemsize*3
            for start in range(0,total,self.chunk):
                if cancelled and cancelled():raise RenderCancelled()
                count=min(self.chunk,total-start);size=count*stride
                self.write(self.buffers[0],size,P(source.ctypes.data+start*stride))
                self.arg(kernel,2,U(count));global_size=S(count)
                _check(self.api.clEnqueueNDRangeKernel(self.queue,kernel,1,None,C.byref(global_size),None,0,None,None))
                _check(self.api.clEnqueueReadBuffer(self.queue,self.buffers[1],1,0,size,P(out.ctypes.data+start*stride),0,None,None))
                if progress:progress(f'GPU 处理 {min(total,start+count)*100//total}%')
            if cancelled and cancelled():raise RenderCancelled()
        return out

    def gr4(self,source,key,tone,exposure=0,amount=1,cancelled=None,progress=None,*,controls=None,input_encoding='srgb'):
        from gr4_accel import config
        from gr4_renderer import exposure_table
        with self.lock:
            cachekey=(key,tone,tuple(sorted((controls or {}).items())),input_encoding)
            if cachekey not in self.configs:
                cfg=config(*cachekey);self.configs[cachekey]=self.buffer(C.sizeof(cfg),C.byref(cfg))
                if len(self.configs)>24:
                    oldest=next(iter(self.configs));handle=self.configs.pop(oldest)
                    _check(self.api.clReleaseMemObject(handle));self.owned.remove(('MemObject',handle))
            else:self.configs[cachekey]=self.configs.pop(cachekey)
            # A 65536-entry exact CPU-generated transfer table avoids GPU pow
            # differences for exposure, without reducing RGB16 input precision.
            table=exposure_table(exposure,input_encoding)
            if not hasattr(self,'exposure_buffer'):self.exposure_buffer=self.buffer(table.nbytes)
            if getattr(self,'exposure_value',None)!=(exposure,input_encoding):
                self.write(self.exposure_buffer,table.nbytes,P(table.ctypes.data));self.exposure_value=(exposure,input_encoding)
            return self.execute(source,'gr4_u16',[P(self.configs[cachekey]),P(self.exposure_buffer),C.c_double(amount)],cancelled,progress)

    def balance(self,source,x,y,cancelled=None):
        name='balance_u8' if source.dtype==np.uint8 else 'balance_u16'
        return self.execute(source,name,[C.c_double(x),C.c_double(y)],cancelled)


def _self_test(device):
    from gr4_renderer import KEYS,tone_count
    from color_balance import transform
    with np.load(ROOT/'gr4-data/verification.npz',allow_pickle=False) as fixture:
        source=fixture['input']
        for key in KEYS:
            for tone in range(tone_count(key)):
                if not np.array_equal(device.gr4(source,key,tone,input_encoding='standard'),fixture[f'{key}__{tone}']):
                    raise ValueError(f'GPU RGB16 validation failed: {key}/{tone}')
    from gr4_renderer import point
    # Dynamic curves and both new input routes must pass on this device too.
    for encoding in ('srgb','linear'):
        controls={'key':1,'contrast':-2,'highlight':2,'shadow':-1,'saturation':3}
        expected=np.floor(point(source.astype(np.float64)/65535,'PositiveFilm',controls=controls,
                               input_encoding=encoding,allow_native=False)*65535+.5).astype(np.uint16)
        actual=device.gr4(source,'PositiveFilm',0,controls=controls,input_encoding=encoding)
        if not np.array_equal(actual,expected):raise ValueError('GPU dynamic colour validation failed')
    # Check both display and full-precision post-filter operations.
    for dtype,maximum in ((np.uint8,255),(np.uint16,65535)):
        a=(source.astype(np.uint32)*maximum//65535).astype(dtype)
        for xy in ((-35,35),(12,-8),(35,-35)):
            expected=np.floor(transform(a.astype(np.float64)/maximum,*xy)*maximum+.5).astype(dtype)
            if not np.array_equal(device.balance(a,*xy),expected):raise ValueError('GPU balance validation failed')


def warmup():
    global _backend,_attempted,_state
    if _mode=='cpu' or _external_busy or os.environ.get('PHOTO_GPU_DISABLE')=='1':return None
    with _lock:
        if _external_busy:return None
        if _attempted:return _backend
        _attempted=True;_state={'state':'checking','device':'','reason':'正在校验显卡计算结果','devices':[]}
        devices=[];candidates=[]
        try:
            api=OpenCL();devices=api.devices()
            _state['devices']=[{k:v for k,v in row.items() if k!='handle'} for row in devices]
            for info in devices:
                if not info['fp64']:continue
                device=None
                try:
                    device=Device(api,info);_self_test(device)
                    sample=np.random.default_rng(481).integers(0,65536,(384,512,3),dtype=np.uint16)
                    start=time.perf_counter();device.gr4(sample,'CrossProcess',1)
                    candidates.append((time.perf_counter()-start,device))
                except Exception as exc:
                    if device:device.close()
                    _state['reason']=type(exc).__name__+': '+str(exc)[:180]
            if candidates:
                candidates.sort(key=lambda item:item[0]);selected=candidates[0][1]
                for _,device in candidates[1:]:device.close()
                from gr4_accel import render_u16
                from color_balance import transform
                sample=np.random.default_rng(482).integers(0,65536,(768,1024,3),dtype=np.uint16)
                start=time.perf_counter();cpu=render_u16(sample,'CrossProcess',1);cpu_gr=time.perf_counter()-start
                start=time.perf_counter();selected.gr4(sample,'CrossProcess',1);gpu_gr=time.perf_counter()-start
                display=(sample//257).astype(np.uint8)
                start=time.perf_counter();transform(display.astype(np.float64)/255,12,-8);cpu_xy=time.perf_counter()-start
                start=time.perf_counter();selected.balance(display,12,-8);gpu_xy=time.perf_counter()-start
                _state.update(state='ready',device=selected.info['name'],reason='GPU 精度检查通过',
                              gr4_enabled=cpu is None or gpu_gr<cpu_gr,balance_enabled=gpu_xy<cpu_xy,
                              benchmark={'cpu_gr':cpu_gr,'gpu_gr':gpu_gr,'cpu_xy':cpu_xy,'gpu_xy':gpu_xy})
                _backend=selected
            else:_state.update(state='cpu',reason=_state['reason'] if _state['reason']!='正在校验显卡计算结果' else '无通过双精度检查的 GPU，使用 CPU')
        except Exception as exc:
            for _,device in candidates:device.close()
            _backend=None
            _state.update(state='cpu',device='',gr4_enabled=False,balance_enabled=False,
                          reason=type(exc).__name__+': '+str(exc)[:180])
        return _backend


def _run(operation,*args,**kwargs):
    global _backend
    if _mode=='cpu' or _external_busy or os.environ.get('PHOTO_GPU_DISABLE')=='1':return None
    device=warmup()
    if device is None:return None
    from native_backend import RenderCancelled
    # Keep dispatch and invalidation atomic: another worker may already have
    # obtained this device when the first worker discovers a driver failure.
    with _lock:
        if _backend is not device or _state.get('state')!='ready':return None
        if _mode=='cpu' or _external_busy or _state.get(operation+'_enabled') is False:return None
        try:return getattr(device,operation)(*args,**kwargs)
        except RenderCancelled:raise
        except Exception:
            # Caller discards any partial GPU result and recomputes all pixels.
            _state.update(state='cpu',reason='GPU 运行失败，已回退 CPU',device='',
                          gr4_enabled=False,balance_enabled=False)
            _backend=None
            device.close()
            return None


def gr4(source,key,tone,exposure=0,amount=1,cancelled=None,progress=None,*,controls=None,input_encoding='srgb'):
    if source.size<4096*3:return None
    return _run('gr4',source,key,tone,exposure,amount,cancelled,progress,controls=controls,input_encoding=input_encoding)


def balance(source,x,y,cancelled=None):
    if source.size<4096*3:return None
    return _run('balance',source,x,y,cancelled)
