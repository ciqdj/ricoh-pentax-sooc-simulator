"""DXGI adapter discovery, saved AI device order, and scoped device failures."""
from pathlib import Path
import ctypes as C
import json
import os
import re
import threading
import time
import uuid

_lock = threading.RLock()
_health = None
_health_path = None
CPU_KEY = 'cpu'


def adapter_kind(adapter):
    """Classify DXGI adapters using their reported memory and device family."""
    vendor = int(adapter.get('vendor', 0))
    dedicated = int(adapter.get('dedicated', 0))
    name = str(adapter.get('name', '')).casefold()
    if vendor == 0x8086:
        return 'discrete' if 'arc' in name and dedicated > 0 else 'integrated'
    if vendor == 0x1002 and any(token in name for token in (
            'radeon(tm) graphics', 'radeon graphics', 'radeon 780m', 'radeon 760m',
            'radeon 680m', 'radeon 660m', 'radeon 610m', 'vega')):
        return 'integrated'
    return 'discrete' if dedicated >= 512 * 1024**2 else 'integrated'


def adapter_key(adapter):
    """Stable, non-personal preference key; the DXGI slot separates twins."""
    return f"gpu:{identity(adapter)}:{int(adapter.get('id', 0))}"


def _priority_path():
    import app_paths
    return Path(app_paths.CONFIG) / 'ai-device-priority.json'


def default_adapter_order(adapters):
    rows = [dict(row, kind=adapter_kind(row), key=adapter_key(row)) for row in adapters]
    return sorted(rows, key=lambda row: (row['kind'] != 'discrete', -row.get('dedicated', 0), row.get('id', 0)))


def normalize_order(adapters, saved_order=None):
    """Merge a saved order with currently detected devices without losing choices."""
    rows = default_adapter_order(adapters)
    gpu_keys = [row['key'] for row in rows]
    default = [*gpu_keys, CPU_KEY]
    if not isinstance(saved_order, list):
        return default
    available = set(default)
    ordered = []
    for key in saved_order:
        if isinstance(key, str) and key in available and key not in ordered:
            ordered.append(key)
    additions = [key for key in default if key != CPU_KEY and key not in ordered]
    ordered.extend(additions)
    if CPU_KEY not in ordered:
        ordered.append(CPU_KEY)
    return ordered


def priority_order(adapters=None):
    rows = adapters if adapters is not None else globals()['adapters']()
    try:
        data = json.loads(_priority_path().read_text(encoding='utf-8'))
        saved = data.get('order') if isinstance(data, dict) and data.get('version') == 1 else None
    except (OSError, ValueError, TypeError, AttributeError):
        saved = None
    return normalize_order(rows, saved)


def unsupported_graphics(rows):
    # AUDIT-P39: report adapters that cannot provide a D3D12 device. Those can
    # never run DirectML, so AI denoising silently falls back to a CPU path
    # that is roughly ten times slower. Say it once instead of letting the
    # user wonder why the status line reads CPU.
    return [row['name'] for row in rows
            if row.get('kind') != 'cpu' and not row.get('d3d12', True)]


def priority_rows(adapters=None):
    rows = default_adapter_order(adapters if adapters is not None else globals()['adapters']())
    by_key = {row['key']: row for row in rows}
    by_key[CPU_KEY] = {'id': None, 'key': CPU_KEY, 'kind': 'cpu', 'name': 'CPU'}
    return [by_key[key] for key in priority_order(rows) if key in by_key]


def save_priority(order, adapters=None):
    rows = adapters if adapters is not None else globals()['adapters']()
    valid = set([row['key'] for row in default_adapter_order(rows)] + [CPU_KEY])
    if (not isinstance(order, (list, tuple)) or
            any(not isinstance(key, str) for key in order) or
            len(order) != len(valid) or set(order) != valid):
        raise ValueError('AI 设备顺序与当前检测到的设备不匹配')
    path = _priority_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump({'version': 1, 'order': list(order)}, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try: temporary.unlink(missing_ok=True)
        except OSError: pass


def d3d12_supported(adapter):
    # AUDIT-P39: DirectML needs a D3D12 device at feature level 11_0 or above.
    # Asking D3D12 directly is exact; guessing from a marketing name is not.
    # A GTX 900 (Maxwell) has it, a GTX 500 (Fermi) does not, and some OEM
    # rebadges share names across both.
    try:
        dll = C.WinDLL('d3d12.dll')
    except OSError:
        return False
    create = dll.D3D12CreateDevice
    create.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.POINTER(C.c_void_p)]
    create.restype = C.c_long
    device = C.c_void_p()
    iid = C.create_string_buffer(uuid.UUID('189819f1-1db6-4b57-be54-1821339b85f7').bytes_le)
    hr = create(adapter, 0xb000, iid, C.byref(device))
    if hr < 0 or not device:
        return False
    try:
        # ID3D12Device::Release is slot 2.
        table = C.cast(device, C.POINTER(C.POINTER(C.c_void_p))).contents
        C.WINFUNCTYPE(C.c_uint, C.c_void_p)(table[2])(device)
    except Exception:
        pass
    return True


def adapters():
    if os.name != 'nt':
        return []
    class Luid(C.Structure):
        _fields_ = [('low', C.c_uint32), ('high', C.c_int32)]
    class Desc(C.Structure):
        _fields_ = [('name', C.c_wchar*128), ('vendor', C.c_uint32), ('device', C.c_uint32),
                    ('subsys', C.c_uint32), ('revision', C.c_uint32), ('dedicated', C.c_size_t),
                    ('system', C.c_size_t), ('shared', C.c_size_t), ('luid', Luid), ('flags', C.c_uint32)]
    class Memory(C.Structure):
        _fields_ = [('budget', C.c_uint64), ('usage', C.c_uint64),
                    ('reservation_available', C.c_uint64), ('reservation', C.c_uint64)]
    def method(obj, index, restype, *args):
        table = C.cast(obj, C.POINTER(C.POINTER(C.c_void_p))).contents
        return C.WINFUNCTYPE(restype, C.c_void_p, *args)(table[index])
    def release(obj):
        if obj: method(obj, 2, C.c_uint)(obj)
    dll = C.WinDLL('dxgi.dll', winmode=0x800)
    factory = C.c_void_p()
    create = dll.CreateDXGIFactory1
    create.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]; create.restype = C.c_long
    iid = C.create_string_buffer(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    if create(iid, C.byref(factory)) < 0:
        raise OSError('DXGI adapter enumeration is unavailable')
    result = []
    try:
        # Match ORT's DMLProviderFactoryCreator::CreateD3D12Device, which calls
        # IDXGIFactory4::EnumAdapters1(device_id) in the installed runtime.
        enum = method(factory, 12, C.c_long, C.c_uint, C.POINTER(C.c_void_p))
        for index in range(32):
            adapter = C.c_void_p()
            if enum(factory, index, C.byref(adapter)) < 0: break
            try:
                desc = Desc()
                if method(adapter, 10, C.c_long, C.POINTER(Desc))(adapter, C.byref(desc)) < 0: continue
                if desc.flags & 2: continue  # DXGI_ADAPTER_FLAG_SOFTWARE.
                row = {'id': index, 'name': desc.name, 'vendor': desc.vendor, 'device': desc.device,
                       'subsys': desc.subsys, 'revision': desc.revision, 'dedicated': desc.dedicated,
                       'shared': desc.shared, 'luid': f'{desc.luid.high & 0xffffffff:08x}:{desc.luid.low:08x}'}
                adapter3 = C.c_void_p()
                iid3 = C.create_string_buffer(uuid.UUID('645967a4-1392-4310-a798-8053ce3e93fd').bytes_le)
                if method(adapter, 0, C.c_long, C.c_void_p, C.POINTER(C.c_void_p))(adapter, iid3, C.byref(adapter3)) >= 0:
                    try:
                        mem = Memory()
                        if method(adapter3, 14, C.c_long, C.c_uint, C.c_uint, C.POINTER(Memory))(adapter3, 0, 0, C.byref(mem)) >= 0:
                            row.update(memory_budget=int(mem.budget), memory_used=int(mem.usage),
                                       memory_available=max(0, int(mem.budget)-int(mem.usage)),
                                       memory_scope='DXGI current-process local budget, not global physical free memory')
                    finally: release(adapter3)
                row['d3d12'] = d3d12_supported(adapter)
                result.append(row)
            finally: release(adapter)
    finally: release(factory)
    seen = set(); unique = []
    for item in result:
        if item['luid'] in seen: continue
        seen.add(item['luid']); unique.append(item)
    return default_adapter_order(unique)


def memory_available():
    if os.name != 'nt': return 0
    class Status(C.Structure):
        _fields_ = [('length', C.c_uint32), ('load', C.c_uint32)] + [(key, C.c_uint64) for key in
            ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'extended')]
    value = Status(); value.length = C.sizeof(value)
    fn = C.windll.kernel32.GlobalMemoryStatusEx; fn.argtypes = [C.POINTER(Status)]; fn.restype = C.c_int
    return int(value.avail_phys) if fn(C.byref(value)) else 0


def identity(adapter):
    # PCI properties are stable across worker restarts and display reordering.
    # Equal physical PCI profiles are conservatively quarantined together on a
    # true device fault; neither display name nor private machine ID is stored.
    return ':'.join(str(adapter.get(k, 0)) for k in ('vendor', 'device', 'subsys', 'revision', 'dedicated'))


def failure(exc):
    text = str(exc).lower()
    codes = sorted(set(re.findall(r'0x[0-9a-f]{8}', text)))
    if type(exc).__name__=='InvalidDenoiseOutput':
        return {'kind':'profile','codes':codes,'reason':'模型输出数值异常；已停用此设备上的该模型配置。',
                'detail':str(exc)}
    if any(code in codes for code in ('0x887a0005', '0x887a0006', '0x887a0007')) or any(word in text for word in ('device removed', 'device hung', 'device reset', 'dxgi_error_device')):
        return {'kind': 'device', 'codes': codes, 'reason': '显卡驱动报告设备停止响应；保留故障保护，不再自动尝试该设备。'}
    if isinstance(exc,MemoryError) or '0x8007000e' in codes or any(word in text for word in ('out of memory', 'e_outofmemory', 'failed to allocate', 'memory allocation', 'bad_alloc', 'not enough memory', 'allocation failed')):
        return {'kind': 'memory', 'codes': codes, 'reason': '此模型分块所需内存不足；保留同一模型，改用较小分块或 CPU。'}
    return {'kind': 'profile', 'codes': codes, 'reason': '此模型的 GPU 执行配置失败；该配置停用，其他已验证配置不受影响。'}


def _path():
    import app_paths
    return Path(app_paths.CONFIG) / 'ai-device-health.json'


def _load():
    global _health, _health_path
    path = _path()
    if _health is None or path != _health_path:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('version') != 1 or not isinstance(data.get('devices'), dict) or not isinstance(data.get('profiles'), dict): raise ValueError()
            for group in ('devices','profiles'):
                for key,item in data[group].items():
                    if not isinstance(key,str) or not isinstance(item,dict) or item.get('kind') not in ('device','memory','profile') or not isinstance(item.get('reason'),str):raise ValueError()
        except FileNotFoundError:
            data = {'version': 1, 'devices': {}, 'profiles': {}}
        except (OSError, ValueError, AttributeError):
            # A damaged health record cannot silently re-enable uncertain GPUs.
            data = {'version': 1, 'devices': {}, 'profiles': {}, 'unreadable': True}
        _health, _health_path = data, path
    return _health


def blocked(adapter, model, variant):
    with _lock:
        data = _load()
        if data.get('unreadable'):
            return {'kind': 'health_record', 'reason': '显卡故障记录无法读取；已保守使用 CPU。'}
        key = identity(adapter)
        return data['devices'].get(key) or data['profiles'].get(f'{key}/{model}/{variant}')


def record_failure(adapter, model, variant, exc):
    item = failure(exc)
    item.update(model=model, variant=variant, recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with _lock:
        data = _load()
        if data.get('unreadable'): return item
        key = identity(adapter)
        target = 'devices' if item['kind'] == 'device' else 'profiles'
        data[target][key if target == 'devices' else f'{key}/{model}/{variant}'] = item
        write_diagnostic(data, _path())
    return item


def write_diagnostic(data, path=None):
    if path is None:
        import app_paths
        path = Path(app_paths.CONFIG) / 'ai-backend-status.json'
    path = Path(path); temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError):
        # Diagnostic storage must not turn a valid inference into a failure.
        pass
    finally:
        try: temporary.unlink(missing_ok=True)
        except OSError: pass


def large_tile_allowed(adapter,tile=384):
    minimum=2*1024**3 if tile<=384 else 4*1024**3
    return adapter.get('dedicated', 0) >= minimum and adapter.get('memory_available', 0) >= minimum
