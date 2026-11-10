"""DXGI properties and scoped AI device failures; never reset GPU drivers.

DirectML device_id is the DXGI EnumAdapters index. Adapter ranking uses actual
dedicated memory, then a measured workload; marketing names are display-only.
"""
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
                result.append(row)
            finally: release(adapter)
    finally: release(factory)
    seen = set(); unique = []
    for item in result:
        if item['luid'] in seen: continue
        seen.add(item['luid']); unique.append(item)
    return sorted(unique, key=lambda item: (-item['dedicated'], item['id']))


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
