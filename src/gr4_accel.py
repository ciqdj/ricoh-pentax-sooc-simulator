"""Optional x64 native CPU kernel. NumPy remains the exact checked fallback."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import ctypes as C
import hashlib,json,os
import numpy as np
from app_paths import ROOT


class Config(C.Structure):
    _fields_=[(name,C.c_double*size) for name,size in (
        ('base',256),('inverse',9),('matrix',9),('saturation',9),('banks',108),('curves',768),('ycc_inverse',9))]+[
        (name,C.c_int32*size) for name,size in (('bounds',16),('selectors',20),('control',21))]+[('input_encoding',C.c_int32)]


@lru_cache(maxsize=1)
def library():
    path=ROOT/'gr4-core/gr4_core.dll'
    if os.name!='nt' or not path.exists() or os.environ.get('GR4_DISABLE_NATIVE')=='1':return None
    manifest=json.loads((ROOT/'gr4-data/manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest.get('native_core_sha256'):raise ValueError('理光加速组件校验失败。')
    dll=C.CDLL(str(path))
    dll.gr4_points.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(Config)];dll.gr4_points.restype=None
    dll.gr4_u16.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(Config),C.c_void_p,C.c_double];dll.gr4_u16.restype=None
    return dll


@lru_cache(maxsize=128)
def config(key,tone,settings=(),input_encoding='srgb'):
    from gr4_renderer import profiles,KEYS,RGB
    from gr4_parameters import calculated,signature
    data=profiles();look=data['looks'][KEYS[key]];cfg=Config()
    overrides=dict(settings)
    if len(look['curves'])>1:overrides['tone']=tone
    curves,matrix=calculated(KEYS[key],signature(KEYS[key],overrides))
    values=dict(base=data['base_gamma'],inverse=data['inverse_standard_matrix'],ycc_inverse=RGB,
                curves=curves,saturation=matrix,**{name:look[name] for name in ('matrix','banks','bounds','selectors','control')})
    for name,value in values.items():
        flat=np.asarray(value).reshape(-1);target=getattr(cfg,name)
        if len(target)!=len(flat):raise ValueError('Invalid native colour table shape')
        target[:]=flat.tolist()
    cfg.input_encoding={'standard':0,'srgb':1,'linear':2}[input_encoding]
    return cfg


def points(rgb,key,tone,*,controls=None,input_encoding='srgb'):
    dll=library()
    if dll is None:return None
    src=np.ascontiguousarray(rgb,dtype=np.float64);out=np.empty_like(src)
    dll.gr4_points(src.ctypes.data,out.ctypes.data,src.size//3,C.byref(config(key,tone,tuple(sorted((controls or {}).items())),input_encoding)))
    return out


def render_u16(rgb,key,tone,cancelled=None,progress=None,*,controls=None,input_encoding='srgb',exposure=0,amount=1):
    from native_backend import RenderCancelled
    dll=library()
    if dll is None:return None
    from gr4_renderer import exposure_table
    src=np.ascontiguousarray(rgb);out=np.empty_like(src);cfg=config(key,tone,tuple(sorted((controls or {}).items())),input_encoding);total=src.size//3
    table=exposure_table(exposure,input_encoding)
    chunk=262144;done=0
    def run(start):
        if cancelled and cancelled():raise RenderCancelled()
        n=min(chunk,total-start)
        dll.gr4_u16(src.ctypes.data+start*6,out.ctypes.data+start*6,n,C.byref(cfg),table.ctypes.data,amount)
        return n
    workers=min(8,max(1,os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(run,start) for start in range(0,total,chunk)]
        try:
            for future in as_completed(pending):
                done+=future.result()
                if cancelled and cancelled():raise RenderCancelled()
                if progress:progress(f'理光 CPU 处理 {done*100//total}%')
        except BaseException:
            for future in pending:future.cancel()
            raise
    return out
