"""GR IV mode controls, using the original integer tone generation.
GRKit native_tone.rs adaptation: MIT, copyright (c) 2026 ahonn.
See gr4-data/LICENSE-GRKit.txt. Camera tables are read from controls.json.
"""
from functools import lru_cache
import hashlib,json
import numpy as np
from app_paths import ROOT

@lru_cache(maxsize=1)
def resources():
    blob=(ROOT/'gr4-data/controls.json').read_bytes()
    manifest=json.loads((ROOT/'gr4-data/manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(blob).hexdigest()!=manifest['controls_sha256']:
        raise ValueError('理光参数资源不完整，请重新取得完整程序。')
    return json.loads(blob)

LABELS={'key':'明暗调','contrast':'对比度','highlight':'高光对比度',
        'shadow':'阴影对比度','saturation':'饱和度','tone':'色调'}
ORDER=('key','contrast','highlight','shadow','saturation','tone')

def defaults(profile):
    return dict(resources()['looks'][profile]['defaults'])

def effective(profile,overrides=None):
    result=defaults(profile)
    overrides={} if overrides is None else overrides
    if not isinstance(overrides,dict) or set(overrides)-set(result):
        raise ValueError('此理光模式不支持所选参数')
    for name,value in overrides.items():
        low,high=(0,2) if name=='tone' else (-4,4)
        if type(value) is not int or not low<=value<=high:
            raise ValueError('理光参数超出范围')
        result[name]=value
    return result

def signature(profile,overrides=None):
    return tuple(sorted(effective(profile,overrides).items()))

def base_curve(tables, key, contrast, shadow, highlight, scale):
    """GRKit integer algorithm; signed division semantics are kept explicit."""
    gain = (tables['gains'][max(-4, min(4, key)) + 4] * 512) >> 8
    u = ((scale * gain) & 0xffffffff) >> 13
    if u < 128:
        u, sample, low, high = 128, 8, 0, 1
    elif u > 2048:
        u, sample, low, high = 2048, 128, 3, 4
    else:
        low = 3 if u >= 1024 else 2 if u >= 512 else 1 if u > 255 else 0
        sample, high = u >> 4, low + 1
    mixes = []
    for step in (contrast * 2 + highlight, contrast * 2, contrast * 2 + shadow):
        if step < -12:
            first, second, weight = 1, 2, 0
        elif step > 12:
            first, second, weight = 0, 1, 512
        elif step >= 1:
            first, second, weight = 0, 1, step * 512 // 12
        else:
            first, second, weight = 1, 2, (step + 12) * 512 // 12
        mixes.append((first, second, weight))

    def at(k, which, i):
        first, second, w = mixes[k]
        family = tables['families']
        return (w * family[first][which][i] + (512 - w) * family[second][which][i] + 256) >> 9

    mid = tables['unit'] * 118
    weights = []
    for k in range(3):
        lo, hi = at(k, low, sample), at(k, high, sample)
        weights.append(512 if lo <= hi else 0 if hi >= mid else ((mid - hi) << 9) // (lo - hi))

    def curve(k, i):
        w = weights[k]
        return (w * at(k, low, i) + (512 - w) * at(k, high, i) + 256) >> 9

    def ease(ratio):
        return tables['ease'][max(0, min(96, ratio - 32))]

    result = []
    for i in range(256):
        inp = 16 * i
        side = curve(2 if u >= inp else 0, i)
        value = curve(1, i)
        if u < inp and highlight:
            if inp >= u << 2:
                value = side
            else:
                w = ease((2 * i << 8) // u)
                value = (value * (2048 - w) + side * w + 1024) >> 11
        if u > inp and shadow:
            if inp <= u >> 2:
                value = side
            else:
                w = ease((u << 5) // inp)
                value = (value * (2048 - w) + side * w + 1024) >> 11
        result.append(min(16384, (value << 14) // (0x3f02 * (tables['unit'] >> 6))))
    return result


def compose(base,record):
    if not record['enabled']:return np.tile(base,(3,1)).astype(np.float64)
    curves=np.asarray(record['curves'],dtype=np.int64)
    weight=min(3,record['weight'])
    mixed=(weight*curves[1]+(3-weight)*curves)//3
    mixed[1]=curves[1]
    code=np.asarray(base,dtype=np.int64)
    i=np.minimum(code>>6,255);j=np.minimum(i+1,255)
    product=(mixed[:,j]-mixed[:,i])*(code&63)
    return (mixed[:,i]+np.sign(product)*(np.abs(product)//64)).astype(np.float64)

@lru_cache(maxsize=128)
def calculated(profile,settings):
    from gr4_renderer import profiles
    data=resources();settings=dict(settings);record=data['looks'][profile]
    tables=data['tables'];scale=data['input_scale']
    neutral=base_curve(tables,0,0,0,0,scale)
    base=base_curve(tables,settings['key'],settings['contrast'],-settings['shadow'],settings['highlight'],scale)
    # Keep the GRKit preview's standard base-gamma coordinates. This same
    # mapping is used by site.rs for its direct linear RAW input tables.
    neutral=np.asarray(neutral,dtype=np.float64)
    gamma=np.asarray(profiles()['base_gamma'],dtype=np.float64)
    i=np.searchsorted(neutral,gamma,side='left');safe=np.clip(i,1,255)
    lo,hi=neutral[safe-1],neutral[safe]
    part=np.divide(gamma-lo,hi-lo,out=np.zeros_like(gamma),where=hi>lo)
    x=np.where(i==0,0,np.where(i==256,255,safe-1+part))
    mapped=np.floor(np.interp(x,np.arange(256),base)+.5).clip(0,16384).astype(np.int64)
    curves=compose(mapped,record['curve_sets'][settings.get('tone',0)])
    look=profiles()['looks'][profile]
    matrix=(np.asarray(record['post_matrices_q9'][str(settings['saturation'])],dtype=np.float64)/512
            if 'saturation' in settings else np.asarray(look['saturation'],dtype=np.float64))
    curves.flags.writeable=False;matrix.flags.writeable=False
    return curves,matrix
