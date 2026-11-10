"""A documented colour-transfer adapter, NOT the complete PENTAX RAW engine.

No photographs or colour patches are fitted. Exact source components are coupled
to an explicitly chosen sRGB adapter. That coupling has not been proven equal to
the original complete pipeline; it must never be advertised as an OEM clone.
"""
import json
import numpy as np
from app_paths import ROOT

DATA=json.loads((ROOT/'data'/'transfer_data.json').read_text(encoding='utf-8'))
LOOKS=DATA['looks']

def srgb_decode(x):
    x=np.asarray(x,dtype=np.float64)
    return np.where(x<=.04045,x/12.92,((np.maximum(x,0)+.055)/1.055)**2.4)

def srgb_encode(x):
    x=np.asarray(x,dtype=np.float64)
    return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,0)**(1/2.4)-.055)

def source_curve(rgb,curves):
    p=np.clip(rgb,0,1)*4095; lo=np.floor(p).astype(np.int32); hi=np.minimum(lo+1,4095); f=p-lo
    out=np.empty_like(p)
    for c in range(3):
        curve=np.asarray(curves[c],dtype=np.float64)/4095
        out[...,c]=curve[lo[...,c]]*(1-f[...,c])+curve[hi[...,c]]*f[...,c]
    return out

def base_curve(rgb,index,inverse=False):
    nodes=np.asarray(DATA['base_curves'][index],dtype=np.float64)
    x,y=nodes[:,0],nodes[:,1]
    if inverse:x,y=y,x
    x,indices=np.unique(x,return_index=True);y=y[indices]
    y[-1]=1.0
    return np.interp(np.clip(rgb,0,1),x,y)

def apply(rgb,look_id,amount=1.0,input_space='srgb'):
    if not 0<=amount<=2: raise ValueError('Amount must be 0..2')
    if look_id not in LOOKS: raise ValueError('Unknown look: '+look_id)
    original=np.asarray(rgb,dtype=np.float64)
    if not np.isfinite(original).all(): raise ValueError('Non-finite input')
    original=np.clip(original,0,1)
    if input_space=='rec709-g24': baseline=srgb_encode(original**2.4)
    elif input_space=='srgb': baseline=original
    else: raise ValueError('Normalize Log/HDR in a colour-managed editor first; input must be srgb or rec709-g24')
    look=LOOKS[look_id]; out=baseline.copy()
    if 'matrix_rgb' in look:
        linear=srgb_decode(out)
        out=srgb_encode(np.clip(linear@np.asarray(look['matrix_rgb']).T,0,1))
    if 'relative_base_curve' in look:
        source,target=look['relative_base_curve']
        out=base_curve(base_curve(out,source,inverse=True),target)
    if 'base_curve_from_linear' in look:
        out=base_curve(srgb_decode(out),look['base_curve_from_linear'])
    if 'curves' in look:
        out=source_curve(out,[DATA['curves'][key] for key in look['curves']])
    out=np.clip(baseline+amount*(out-baseline),0,1)
    if input_space=='rec709-g24': out=srgb_decode(out)**(1/2.4)
    return out
