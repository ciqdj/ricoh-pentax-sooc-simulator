"""Explicit eight-colour HSL edits, separate from the original camera engine.

Neutral settings preserve every input bit. Named-colour controls exclude the
neutral axis, and taper continuously near it, so numerical hue fluctuations
in grey pixels cannot cause large lightness changes. All arithmetic is
float64 until the single final quantization to the original RGB8/RGB16 type.
"""
import math
import numpy as np

COLORS=('red','orange','yellow','green','aqua','blue','purple','magenta')
NAMES=('红','橙','黄','绿','青','蓝','紫','洋红')
SWATCHES=('#ed6469','#eea35b','#d8c969','#79bc7a','#6abfbd','#688fda','#aa84d0','#d581bc')
FIELDS=('hue','saturation','luminance')
DEFAULT={key:{field:0. for field in FIELDS} for key in COLORS}
CENTRES=np.array((0.,30.,60.,120.,180.,240.,270.,300.,360.))
# Editing-domain colour discrimination, not a camera calibration. Smoothstep
# reaches full named-colour weight at a 5% RGB channel separation. Its zero
# slope at the neutral axis avoids introducing a hard near-grey boundary.
_FULL_COLOUR_CHROMA=0.05


def validate(value=None):
    value={} if value is None else value
    if not isinstance(value,dict) or set(value)-set(COLORS):raise ValueError('混色器颜色无效')
    result={}
    for color in COLORS:
        setting=value.get(color,{})
        if not isinstance(setting,dict) or set(setting)-set(FIELDS):raise ValueError('混色器参数无效')
        result[color]={}
        for field in FIELDS:
            number=setting.get(field,0.)
            if type(number) not in (float,int) or not math.isfinite(number) or not -100<=number<=100:raise ValueError('混色器范围为 −100 至 +100')
            result[color][field]=float(number)
    return result


def neutral(value=None):return not any(v for settings in validate(value).values() for v in settings.values())


def _adjust(rgb,values):
    maximum=np.max(rgb,axis=-1);minimum=np.min(rgb,axis=-1);delta=maximum-minimum
    light=(maximum+minimum)/2
    sat=np.divide(delta,1-np.abs(2*light-1),out=np.zeros_like(delta),where=(delta>1e-12))
    hue=np.zeros_like(delta);safe=np.where(delta>1e-12,delta,1.)
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    red=(maximum==r)&(delta>1e-12);green=(maximum==g)&~red&(delta>1e-12);blue=~red&~green&(delta>1e-12)
    hue[red]=((g-b)/safe)[red]%6;hue[green]=((b-r)/safe+2)[green];hue[blue]=((r-g)/safe+4)[blue]
    hue*=60
    left=np.clip(np.searchsorted(CENTRES,hue,side='right')-1,0,7);right=(left+1)%8
    fraction=(hue-CENTRES[left])/(CENTRES[left+1]-CENTRES[left]);weight=fraction*fraction*(3-2*fraction)
    controls=values[left]*(1-weight[...,None])+values[right]*weight[...,None]
    # A smooth partition between adjacent named colours avoids hard hue masks.
    colour=np.clip(delta/_FULL_COLOUR_CHROMA,0.,1.)
    colour=colour*colour*(3.-2.*colour)
    controls*=colour[...,None]
    hue=(hue+controls[...,0]*.3)%360
    sat=np.clip(sat*(1+controls[...,1]/100),0,1)
    gain=np.exp2(controls[...,2]/50)
    # Endpoints remain anchored instead of clipping an entire highlight band.
    light=light*gain/(1+light*(gain-1))
    chroma=(1-np.abs(2*light-1))*sat;sector=hue/60
    second=chroma*(1-np.abs(sector%2-1));offset=light-chroma/2
    index=np.floor(sector).astype(np.int8);zero=np.zeros_like(chroma)
    out=np.stack((np.select((index==0,index==1,index==2,index==3,index==4,index==5),(chroma,second,zero,zero,second,chroma)),
                  np.select((index==0,index==1,index==2,index==3,index==4,index==5),(second,chroma,chroma,second,zero,zero)),
                  np.select((index==0,index==1,index==2,index==3,index==4,index==5),(zero,zero,second,chroma,chroma,second))),axis=-1)
    return np.clip(out+offset[...,None],0,1)


def apply(rgb,value=None,cancelled=None):
    setting=validate(value)
    if not isinstance(rgb,np.ndarray) or rgb.dtype not in (np.uint8,np.uint16) or rgb.ndim!=3 or rgb.shape[-1]!=3 or not all(rgb.shape[:2]):raise ValueError('混色器需要非空 RGB8 / RGB16')
    if cancelled and cancelled():
        from native_backend import RenderCancelled
        raise RenderCancelled()
    if not any(v for group in setting.values() for v in group.values()):return rgb
    values=np.array([[setting[c][f] for f in FIELDS] for c in COLORS],dtype=np.float64)
    maximum=np.iinfo(rgb.dtype).max;out=np.empty_like(rgb)
    for y in range(0,rgb.shape[0],64):
        if cancelled and cancelled():
            from native_backend import RenderCancelled
            raise RenderCancelled()
        out[y:y+64]=np.floor(_adjust(rgb[y:y+64].astype(np.float64)/maximum,values)*maximum+.5).astype(rgb.dtype)
    if cancelled and cancelled():
        from native_backend import RenderCancelled
        raise RenderCancelled()
    return out
