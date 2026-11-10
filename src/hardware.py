"""DXGI adapter order, used only to select the matching DirectML device."""
import ctypes as C
import uuid


def adapters():
    class Luid(C.Structure):_fields_=[('low',C.c_uint32),('high',C.c_int32)]
    class Desc(C.Structure):
        _fields_=[('name',C.c_wchar*128),('vendor',C.c_uint32),('device',C.c_uint32),('subsys',C.c_uint32),('revision',C.c_uint32),
                  ('dedicated',C.c_size_t),('system',C.c_size_t),('shared',C.c_size_t),('luid',Luid),('flags',C.c_uint32)]
    def method(obj,index,restype,*argtypes):
        table=C.cast(obj,C.POINTER(C.POINTER(C.c_void_p))).contents
        return C.WINFUNCTYPE(restype,C.c_void_p,*argtypes)(table[index])
    def release(obj):method(obj,2,C.c_uint)(obj)
    factory=C.c_void_p();dll=C.WinDLL('dxgi.dll',winmode=0x800)
    create=dll.CreateDXGIFactory1;create.argtypes=[C.c_void_p,C.POINTER(C.c_void_p)];create.restype=C.c_long
    iid=C.create_string_buffer(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    code=create(iid,C.byref(factory))
    if code<0:raise OSError('DXGI unavailable')
    rows=[]
    try:
        enum=method(factory,12,C.c_long,C.c_uint,C.POINTER(C.c_void_p))
        for i in range(16):
            adapter=C.c_void_p();code=enum(factory,i,C.byref(adapter))
            if code<0:break
            try:
                desc=Desc();code=method(adapter,10,C.c_long,C.POINTER(Desc))(adapter,C.byref(desc))
                if code>=0 and not (desc.flags&2):
                    rows.append({'id':i,'name':desc.name,'vendor':desc.vendor,'dedicated':desc.dedicated,'shared':desc.shared})
            finally:release(adapter)
    finally:release(factory)
    return rows
