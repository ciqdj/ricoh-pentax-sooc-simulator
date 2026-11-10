"""Version-locked original GUI defaults and original parameter setters.

The defaults come from DCU's own CCustomImageParam initializer, not a list of
hand-tuned adjustments. See 0095b0 -> 06c910, 0090e0, and 007370 in DCU 5.10.2.2.
All objects are private to the disposable native worker.
"""
import ctypes as C


FIELDS = {'saturation': 0x08, 'hue': 0x0c, 'key': 0x10,
          'contrast': 0x14, 'highlight': 0x1c, 'shadow': 0x20,
          'sharpness': 0x24, 'filter_effect': 0x2c,
          'mono_toning': 0x30, 'toning': 0x34}

# DCU 5.10.2.2's original Noise Reduction panel (201470) sets these
# trackbar and numeric-entry ranges. 202dd0 maps those widgets to the
# CCustomImageParam members below; 007370 calls the original SDK setters.
# These controls are independent of the selected Custom Image colour mode.
# Presence in overrides means an explicit enabled setting. Absence preserves
# the input's original setting, including its enable flag, without any write.
NR_CONTROLS = {
    'nr_random_noise': {'name_zh': '去除随机杂点', 'min': 0, 'max': 200, 'default': 0,
                        'tooltip': 'DCU 原厂数值 0–200，非百分比；作用随原厂参数非线性变化。'},
    'nr_noise_refinement': {'name_zh': '杂点纯净化', 'min': 0, 'max': 100, 'default': 0,
                            'tooltip': 'DCU 原厂数值 0–100，非百分比；与随机杂点参数分别控制。'},
    'nr_false_color': {'name_zh': '抑制伪色彩信号', 'min': 0, 'max': 256, 'default': 80,
                       'tooltip': 'DCU 原厂数值 0–256，非百分比；恢复时使用本图的原有设置。'},
    'nr_defringe': {'name_zh': '去除边缘', 'min': 0, 'max': 100, 'default': 0,
                    'tooltip': 'DCU 原厂数值 0–100，非百分比；用于原厂边缘处理。'},
}
# Help/detail text for compatibility inputs; not a claim that a Pentax noise
# calibration describes the source camera's sensor. Never copy another brand's
# ISO into a Pentax calibration silently to make the controls appear stronger.
NR_COMPATIBILITY_HELP = (
    '跨品牌照片通过 RGB 兼容输入处理，当前兼容输入的原厂噪声参数按 ISO 100 生成。'
    '这不等同于原相机的 RAW 噪声校准；滑块保留 DCU 原始数值，不按百分比换算。'
)
NR_FIELDS = {
    'nr_random_noise': (0x1a8, 0x1ac, 0x12c),
    'nr_noise_refinement': (0x1b0, 0x1b4, 0x130),
    'nr_false_color': (0x1b8, 0x1bc, 0x128),
    'nr_defringe': (0x1c0, 0x1c4, 0x288),
}


def available(mode):
    """Controls that original 007370 actually sends for this mode."""
    keys = ['sharpness']
    if mode not in (19, 23): keys += ['saturation']
    if mode not in (19, 23, 24, 20, 29): keys += ['hue']
    if mode != 23: keys += ['key', 'contrast']
    if mode not in (23, 24, 255): keys += ['highlight', 'shadow']
    if mode == 19: keys += ['filter_effect', 'mono_toning']
    if mode in (20, 24, 29, 255): keys += ['toning']
    return keys


def validate_overrides(mode, values):
    if not isinstance(values,dict):raise ValueError('无效的原厂调整参数')
    for key,value in values.items():
        if type(value) is not int:
            raise ValueError('无效的原厂调整参数')
        if key in NR_CONTROLS:
            spec = NR_CONTROLS[key]
            lo,hi = spec['min'],spec['max']
        elif key in available(mode):
            lo,hi=(0,8) if key in ('filter_effect','toning') else (-4,4)
        else:
            raise ValueError('无效的原厂调整参数')
        if not lo<=value<=hi:raise ValueError('原厂调整参数超出范围')


def _noise_record(custom, ui=None):
    record = {}
    for key, (enabled_offset, value_offset, ui_offset) in NR_FIELDS.items():
        item = {'enabled': bool(C.c_int.from_address(custom + enabled_offset).value),
                'value': C.c_int.from_address(custom + value_offset).value}
        if ui is not None:
            item['ui_value'] = C.c_int.from_address(ui + ui_offset).value
        record[key] = item
    return record


def _record(custom, table, mode):
    buckets = C.c_uint64.from_address(custom + table + 8).value
    count = C.c_uint32.from_address(custom + table + 16).value
    if not buckets or not 0 < count < 10000:
        raise RuntimeError('原厂模式参数表未初始化')
    code = mode >> 4
    node = C.c_uint64.from_address(buckets + (code % count) * 8).value
    for _ in range(256):
        if not node:
            break
        if (C.c_int.from_address(node).value == mode and
                C.c_uint32.from_address(node + 24).value == code):
            return C.c_uint64.from_address(node + 8).value
        node = C.c_uint64.from_address(node + 16).value
    raise RuntimeError('原厂参数表缺少所选模式')


def initialize(host, source, ui, keep, mode, exposure=0.0, overrides=None,
               factory=True):
    custom_storage = C.create_string_buffer(0x800)
    extra_storage = C.create_string_buffer(0x100)
    keep.extend((custom_storage, extra_storage))
    custom = C.addressof(custom_storage)
    extra = C.addressof(extra_storage)
    host.call(0x32d0, C.c_void_p, [C.c_void_p], custom)
    text = C.create_unicode_buffer(str(source)); keep.append(text)
    host.call(0x19d60, None, [C.c_void_p, C.c_void_p, C.c_int],
              custom + 8, C.addressof(text), len(str(source).encode('utf-16le')) // 2)
    host.call(0x95b0, None, [C.c_void_p], custom)
    if mode == -1:
        mode = C.c_int.from_address(custom + 0x348).value
    active = host.call(0x90e0, C.c_void_p, [C.c_void_p, C.c_int], custom, mode)
    if not active:
        raise RuntimeError('无法选择原厂模式')
    if factory:
        original = _record(custom, 0x310, mode)
        host.call(0x31c0, None, [C.c_void_p, C.c_void_p], active, original)
    defaults = {key: C.c_int.from_address(active + offset).value
                for key, offset in FIELDS.items()}
    noise_defaults = _noise_record(custom)
    default_highlight_enabled=bool(C.c_int.from_address(active + 0x18).value)
    overrides = overrides or {}
    validate_overrides(mode,overrides)
    for key, value in overrides.items():
        if key in NR_FIELDS:
            enabled_offset, value_offset, _ = NR_FIELDS[key]
            C.c_int.from_address(custom + enabled_offset).value = 1
            C.c_int.from_address(custom + value_offset).value = value
        else:
            C.c_int.from_address(active + FIELDS[key]).value = value
    if 'highlight' in overrides or 'shadow' in overrides:
        C.c_int.from_address(active + 0x18).value = 1
    host.call(0xa9a2a0, C.c_void_p, [C.c_void_p], extra)
    active_extra = C.c_int(0); keep.append(active_extra)
    host.call(0x7370, None, [C.c_void_p] * 4 + [C.c_int],
              custom, ui, extra, C.addressof(active_extra), 0)
    # Exposure belongs to the original UI parameter object; even 0 EV uses the
    # original GUI's explicit-zero branch (007370 -> a937f0).
    host.call(0xa937f0, C.c_int, [C.c_void_p, C.c_int, C.c_double],
              ui, 1, exposure)
    return {'factory_defaults': defaults,
            'factory_noise_reduction': noise_defaults,
            'effective_noise_reduction': _noise_record(custom, ui),
            'factory_highlight_shadow_enabled':default_highlight_enabled,
            'native_mode': mode,
            'available_controls': available(mode),
            'highlight_shadow_enabled': bool(C.c_int.from_address(active + 0x18).value),
            'effective_controls': {key: C.c_int.from_address(active + offset).value
                                   for key, offset in FIELDS.items()},
            'additional_controls_active': bool(active_extra.value),
            'extra_bytes': C.string_at(extra, 0x44).hex(),
            'ui_bytes': C.string_at(ui, 0x2c8).hex()}, extra
