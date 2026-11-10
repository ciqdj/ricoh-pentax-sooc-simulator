"""GR IV colour stages with explicit input domains and native tone controls.

Port of the unchanged R2 JavaScript renderer. GRKit preview mathematics are
MIT licensed, copyright (c) 2026 ahonn; see gr4-data/LICENSE-GRKit.txt.
Native-derived tables are used as recorded, without photographic fitting.
"""
from functools import lru_cache
import hashlib
import json
import numpy as np
from app_paths import ROOT

KEYS = {'Standard':'standard','Vivid':'vivid','MonotoneGR':'mono',
        'SoftMonotone':'softmono','HardMonotone':'hardmono','HighContrastBW':'highmono',
        'PositiveFilm':'positive','CinemaYellow':'yellow','CinemaGreen':'green',
        'NegativeFilm':'negative','CrossProcess':'cross'}
MONO = ('MonotoneGR','SoftMonotone','HardMonotone','HighContrastBW')
YCC = np.array([77/256,150/256,29/256,-43/256,-85/256,128/256,128/256,-107/256,-21/256])
K = 7.8/.18


def inverse(m):
    a,b,c,d,e,f,g,h,i=m
    cof=np.array([e*i-f*h,c*h-b*i,b*f-c*e,f*g-d*i,a*i-c*g,c*d-a*f,d*h-e*g,b*g-a*h,a*e-b*d])
    return cof/(a*cof[0]+b*cof[3]+c*cof[6])


RGB=inverse(YCC)


def mul(m,v):
    """Preserve the scalar JS evaluation order; no BLAS or FMA dependency."""
    m=np.asarray(m);r,g,b=v[...,0],v[...,1],v[...,2]
    return np.stack((m[...,0]*r+m[...,1]*g+m[...,2]*b,
                     m[...,3]*r+m[...,4]*g+m[...,5]*b,
                     m[...,6]*r+m[...,7]*g+m[...,8]*b),axis=-1)


@lru_cache(maxsize=1)
def profiles():
    folder=ROOT/'gr4-data'
    blob=(folder/'profiles.json').read_bytes()
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(blob).hexdigest()!=manifest['profiles_sha256']:
        raise ValueError('理光滤镜资源校验失败，请重新取得完整程序。')
    data=json.loads(blob)
    if set(data['looks'])!=set(KEYS.values()):raise ValueError('理光滤镜资源不完整。')
    return data


def default_tone(key):
    if key not in KEYS:raise ValueError('此 GR4 模式尚未开放处理。')
    return int(profiles()['looks'][KEYS[key]]['default_tone'])


def tone_count(key):
    if key not in KEYS:raise ValueError('此 GR4 模式尚未开放处理。')
    return len(profiles()['looks'][KEYS[key]]['curves'])


def validate_controls(key,controls=None):
    return control_values(key,controls).get('tone',0)


def control_values(key,controls=None):
    from gr4_parameters import effective
    if key not in KEYS:raise ValueError('此 GR4 模式尚未开放处理。')
    return effective(KEYS[key],controls)


def input_domain(value):
    if value not in ('linear','srgb','standard'):raise ValueError('未知的理光输入色彩编码')
    return value


def decode_srgb(rgb):
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def gain(y):
    y=np.clip(y*1023,0,1023)
    starts=np.array([0,512,512,432]);slopes=np.array([73,0,-7,-14]);breaks=np.array([0,56,768,856])
    idx=np.searchsorted(breaks,y,side='right')-1
    return np.clip(starts[idx]+slopes[idx]*(y-breaks[idx])/8,0,1023)/512


def undisplay(rgb):
    ycc=mul(YCC,rgb);g=np.maximum(.25,gain(ycc[...,0]))
    scaled=ycc.copy();scaled[...,1:]/=g[...,None]
    return np.where((g==1)[...,None],rgb,mul(RGB,scaled))


def display(rgb):
    ycc=mul(YCC,rgb);g=gain(ycc[...,0])
    scaled=ycc.copy();scaled[...,1:]*=g[...,None]
    return np.clip(np.where((g==1)[...,None],rgb,mul(RGB,scaled)),0,1)


def undo_curve(curve,v):
    curve=np.asarray(curve);v=np.clip(v,curve[0],curve[255])
    idx=np.minimum(254,np.searchsorted(curve[1:],v,side='left'))
    a,b=curve[idx],curve[idx+1]
    part=np.divide(v-a,b-a,out=np.zeros_like(v),where=b>a)
    return (idx+np.clip(part,0,1))/K


def lookup(curve,x):
    curve=np.asarray(curve);x=np.clip(x,0,255);idx=np.minimum(254,np.floor(x).astype(np.int32));t=x-idx
    return curve[idx]*(1-t)+curve[idx+1]*t


def hue(rgb,look):
    ycc=mul(YCC,rgb);y,cb,cr=ycc[...,0],ycc[...,1],ycc[...,2]
    p=cr>=0;q=np.where(p,np.where(cb>=0,0,3),np.where(cb>=0,1,2))
    a=np.where(p,np.where(cb>=0,cr,-cb),np.where(cb>=0,cb,-cr))
    b=np.where(p,np.where(cb>=0,cb,cr),np.where(cb>=0,-cr,-cb))
    neutral=a+b<=1e-12
    coord=q*2048+np.divide(2048*b,a+b,out=np.zeros_like(a),where=~neutral)
    quarter=np.minimum(3,np.floor(coord/2048).astype(np.int32));quadrant=np.array([0,3,2,1])[quarter]
    x=np.floor(coord-2048*quarter);bounds=np.asarray(look['bounds']);slot=np.zeros_like(quarter);pending=np.ones_like(neutral)
    for s in range(4):
        pending=pending&(x<bounds[quadrant*4+s]);slot+=pending
    bank=np.asarray(look['selectors'])[quadrant*5+slot]
    c=np.asarray(look['control']);yy=np.clip(3000*y,0,4095);breaks=np.array([0,c[18],c[19],c[20]])
    k=np.full_like(quarter,3)
    for s in (3,2,1):k=np.where((k==s)&(yy<breaks[s]),s-1,k)
    slope=c[14+k];slope=np.where((slope&0x100)!=0,slope-0x200,slope)
    weight=np.clip((c[10+k]+slope*(yy-breaks[k])/32)/1024,0,1)
    mixed=mul(np.asarray(look['banks'])[bank],rgb)
    return np.where(neutral[...,None],rgb,np.maximum(0,rgb+weight[...,None]*(mixed-rgb)))


def point(rgb,key,tone=None,*,controls=None,input_encoding='srgb',allow_native=True):
    data=profiles()
    settings=dict(controls or {})
    if tone is not None and tone_count(key)>1:settings['tone']=tone
    elif tone is not None and tone!=0:raise ValueError('无效的色调。')
    settings=control_values(key,settings);encoding=input_domain(input_encoding)
    rgb=np.asarray(rgb,dtype=np.float64)
    if rgb.shape[-1]!=3 or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:raise ValueError('像素必须是 0–1 范围的 RGB。')
    if allow_native:
        from gr4_accel import points
        accelerated=points(rgb,key,settings.get('tone',0),controls=settings,input_encoding=encoding)
        if accelerated is not None:return accelerated
    look=data['looks'][KEYS[key]]
    from gr4_parameters import calculated
    curves,matrix=calculated(KEYS[key],tuple(sorted(settings.items())))
    if encoding=='standard':
        codes=np.clip(undisplay(rgb),0,1)
        linear=mul(data['inverse_standard_matrix'],undo_curve(data['base_gamma'],codes*16384))
    else:linear=rgb if encoding=='linear' else decode_srgb(rgb)
    linear=hue(mul(look['matrix'],linear),look)
    mixed=np.stack([lookup(curves[i],K*linear[...,i])/16384 for i in range(3)],axis=-1)
    return display(mul(matrix,mixed))


@lru_cache(maxsize=8)
def exposure_table(exposure,input_encoding='srgb'):
    from color_math import srgb_encode
    base=np.arange(65536,dtype=np.float64)/65535
    if not exposure:return base
    if input_domain(input_encoding)=='linear':return np.clip(base*2**exposure,0,1)
    linear=decode_srgb(base)
    return np.clip(srgb_encode(linear*2**exposure),0,1)


def render(rgb,key,controls=None,exposure=0,amount=1,cancelled=None,progress=None,*,allow_gpu=True,input_encoding='srgb'):
    from native_backend import RenderCancelled
    from color_math import srgb_encode
    tone=validate_controls(key,controls)
    encoding=input_domain(input_encoding)
    if rgb.dtype!=np.uint16 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('需要 RGB16 输入。')
    if not np.isfinite(exposure) or not -3<=exposure<=3 or not np.isfinite(amount) or not 0<=amount<=2:raise ValueError('浓淡或曝光超出范围。')
    if allow_gpu:
        from gpu_compute import gr4 as gpu_render
        accelerated=gpu_render(rgb,key,tone,exposure,amount,cancelled,progress,controls=controls,input_encoding=encoding)
        if accelerated is not None:return accelerated
    from gr4_accel import render_u16
    accelerated=render_u16(rgb,key,tone,cancelled,progress,controls=controls,input_encoding=encoding,exposure=exposure,amount=amount)
    if accelerated is not None:return accelerated
    transfer=exposure_table(exposure,encoding)
    result=np.empty_like(rgb)
    for row in range(0,len(rgb),48):
        if cancelled and cancelled():raise RenderCancelled()
        raw=rgb[row:row+48];base=raw.astype(np.float64)/65535
        if encoding=='linear':base=np.clip(srgb_encode(base),0,1)
        if amount==0:
            result[row:row+48]=np.floor(base*65535+.5).astype(np.uint16);continue
        processed=point(transfer[raw],key,controls=controls,input_encoding=encoding)
        if amount!=1:processed=np.clip(base+(processed-base)*amount,0,1)
        result[row:row+48]=np.floor(processed*65535+.5).astype(np.uint16)
        if progress:progress(f'理光色彩处理 {min(len(rgb),row+48)*100//len(rgb)}%')
    return result
