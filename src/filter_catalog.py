"""Brand-scoped catalog. A recovered parameter record is not a working filter."""
from native_backend import LOOKS, SEASONS
from engine import LOOKS as LEGACY_ALL
from gr4_renderer import KEYS as GR_AVAILABLE

PENTAX = 'pentax'
RICOH = 'ricoh_gr4'
CUSTOM = 'custom'
BRANDS = {PENTAX: '宾得', RICOH: '理光 GR4', CUSTOM: 'LUT 滤镜'}
CREATIVE_LOOKS = {k:dict(v,origin='legacy31') for k,v in LEGACY_ALL.items()
                  if k.startswith(('Mono','CP_')) or k in ('BaseTone','ReversalTone')}
PENTAX_LOOKS = {**LOOKS, **CREATIVE_LOOKS}
PENTAX_GROUPS = {'全部': (*SEASONS, *(k for k in LOOKS if k not in SEASONS),*CREATIVE_LOOKS),
                 '四季': SEASONS,
                 '彩色': tuple(k for k in LOOKS if k not in SEASONS and k != 'Monochrome'),
                 '黑白': ('Monochrome', *(k for k in CREATIVE_LOOKS if k.startswith('Mono'))),
                 '创意组合': tuple(k for k in CREATIVE_LOOKS if k.startswith('CP_')),
                 '基调曲线': ('BaseTone','ReversalTone')}
# Names are recovered firmware enum names, not invented render presets.
# Three remaining branches are listed honestly, never mapped to another look.
GR_LOOKS = {
    'Standard': {'name': '标准', 'config': 11},
    'Vivid': {'name': '鲜明', 'config': 12},
    'MonotoneGR': {'name': '黑白', 'config': 13},
    'SoftMonotone': {'name': '柔和黑白', 'config': 14},
    'HardMonotone': {'name': '硬调黑白', 'config': 15},
    'HighContrastBW': {'name': '高对比黑白', 'config': 16},
    'PositiveFilm': {'name': '正片', 'config': 17},
    'CinemaYellow': {'name': '电影黄调', 'config': 18},
    'CinemaGreen': {'name': '电影绿调', 'config': 19},
    'BleachBypass': {'name': '漂白旁路', 'config': 20},
    'Retro': {'name': '复古', 'config': 21},
    'HdrCuisine': {'name': 'HDR 调色', 'config': 22},
    'CrossProcess': {'name': '交叉处理', 'config': 23},
    'NegativeFilm': {'name': '负片', 'config': 24},
}
GR_MONO = ('MonotoneGR', 'SoftMonotone', 'HardMonotone', 'HighContrastBW')
GR_GROUPS = {'全部': tuple(GR_LOOKS),
             '彩色': tuple(k for k in GR_LOOKS if k not in GR_MONO),
             '黑白': GR_MONO}
GR_UNAVAILABLE = '此 GR4 模式尚未完成照片处理，暂不能预览或导出。'
GR_NOTICE = 'GR4 色彩研究模型 · 11 种模式可用；完整机内处理一致性尚未验证。'
GR_ENGLISH = {'MonotoneGR':'Monotone','SoftMonotone':'Soft Monotone','HardMonotone':'Hard Monotone',
              'HighContrastBW':'Hi-Contrast B&W','PositiveFilm':'Positive Film','CinemaYellow':'Cinema Yellow',
              'CinemaGreen':'Cinema Green','BleachBypass':'Bleach Bypass','HdrCuisine':'HDR Tone',
              'CrossProcess':'Cross Processing','NegativeFilm':'Negative Film'}


def catalog(brand):
    if brand == PENTAX: return PENTAX_LOOKS
    if brand == RICOH: return GR_LOOKS
    if brand == CUSTOM:
        from custom_lut import get_library
        return get_library().catalog()
    raise ValueError('Unknown filter brand')


def groups(brand):
    if brand == PENTAX: return PENTAX_GROUPS
    if brand == RICOH: return GR_GROUPS
    if brand == CUSTOM:return {'全部':tuple(catalog(CUSTOM))}
    raise ValueError('Unknown filter brand')


def renderer_available(brand,key):
    if brand==CUSTOM:
        from custom_lut import get_library
        try:get_library().lookup(key)
        except (ValueError,KeyError):return False
        return True
    return key in (PENTAX_LOOKS if brand==PENTAX else GR_AVAILABLE if brand==RICOH else {})


def require_renderer(brand,key=None):
    if brand not in BRANDS:raise ValueError('Unknown filter brand')
    if key is not None and not renderer_available(brand,key):raise ValueError(GR_UNAVAILABLE if brand==RICOH else '自定义滤镜不存在' if brand==CUSTOM else '未知宾得滤镜')
