"""Version-locked, headless native constructor probe in its own process.

Never starts DCU's WinMain, invokes UI APIs, injects into the user's app, writes
the installed image, or changes colour computations. Startup is restricted to the inspected CRT
and native engine initialization functions. Output equivalence is still under test.
"""
from pathlib import Path
import ctypes as C, hashlib, json, os, struct
import pefile

import sys
RESOURCE_ROOT=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))
EXE=RESOURCE_ROOT/'native'/'Digital Camera Utility 5.exe'
if not EXE.is_file() and not getattr(sys,'frozen',False):
    EXE=Path(os.environ.get('ProgramFiles','C:/Program Files'))/'PENTAX/Digital Camera Utility 5/Digital Camera Utility 5.exe'
EXPECTED='d55ec396ce9c93a3e61e83e2ed91f774787a16f32de9ea3a9cdd4fee9e75d387'


def emit(event,**fields):
    if sys.stdout is not None:print(json.dumps({'event':event,**fields}),flush=True)

class Host:
    def __init__(self):
        if hashlib.file_digest(EXE.open('rb'),'sha256').hexdigest()!=EXPECTED:
            raise RuntimeError('原厂引擎版本不匹配，请重新取得完整安装包')
        if getattr(sys,'frozen',False):
            manifest=json.loads((RESOURCE_ROOT/'native-data/engine-manifest.json').read_text(encoding='utf-8'))
            for name,expected in manifest.items():
                with (EXE.parent/name).open('rb') as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                        raise RuntimeError('原厂配套组件不完整，请重新取得完整安装包')
        self.k=C.WinDLL('kernel32',use_last_error=True)
        self.k.SetErrorMode(0x0001|0x0002|0x8000)
        self.k.LoadLibraryW.argtypes=[C.c_wchar_p];self.k.LoadLibraryW.restype=C.c_void_p
        self.k.GetProcAddress.argtypes=[C.c_void_p,C.c_void_p];self.k.GetProcAddress.restype=C.c_void_p
        self.k.VirtualProtect.argtypes=[C.c_void_p,C.c_size_t,C.c_uint32,C.POINTER(C.c_uint32)]
        self.k.VirtualProtect.restype=C.c_int
        self.base=self.k.LoadLibraryW(str(EXE))
        if not self.base:raise C.WinError(C.get_last_error())
        # Only headers and imports are consumed by this loader. Parsing all
        # resources/debug/unwind data cost ~15 seconds on every preview.
        self.pe=pefile.PE(str(EXE),fast_load=True)
        self.pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
        # The original XMP initializer loads its companion by basename. Preload
        # the installed companion using its absolute path inside this process.
        companion=EXE.parent/'IslXSdkXMPDLL.dll'
        companion_module=self.k.LoadLibraryW(str(companion))
        if not companion_module:raise C.WinError(C.get_last_error())
        count=0;self.import_slots={};self.callbacks=[];self.blocked_ui=[]
        for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
            name=entry.dll.decode('ascii')
            private=EXE.parent/name
            path=private if private.is_file() else Path(os.environ['SystemRoot'])/'System32'/name
            module=self.k.LoadLibraryW(str(path))
            if not module:raise C.WinError(C.get_last_error())
            for item in entry.imports:
                lookup=C.cast(C.c_char_p(item.name),C.c_void_p) if item.name else C.c_void_p(item.ordinal)
                proc=self.k.GetProcAddress(module,lookup)
                if not proc:raise RuntimeError(f'Unresolved import {name}:{item.name or item.ordinal}')
                address=self.base+item.address-self.pe.OPTIONAL_HEADER.ImageBase
                old=C.c_uint32()
                if not self.k.VirtualProtect(address,8,4,C.byref(old)):raise C.WinError(C.get_last_error())
                C.c_uint64.from_address(address).value=proc
                ignored=C.c_uint32();self.k.VirtualProtect(address,8,old.value,C.byref(ignored))
                count+=1
                if item.name:self.import_slots[item.name.decode('ascii')]=address
        emit('mapped_imports',count=count,base=hex(self.base),gui_entry_called=False)
        self._exception_callback=C.WINFUNCTYPE(C.c_long,C.c_void_p)(self._exception)
        self.k.AddVectoredExceptionHandler.argtypes=[C.c_uint32,C.c_void_p]
        self.k.AddVectoredExceptionHandler.restype=C.c_void_p
        self.k.AddVectoredExceptionHandler(1,C.cast(self._exception_callback,C.c_void_p))

    def _exception(self,info):
        try:
            record,context=struct.unpack('<QQ',C.string_at(info,16))
            code=C.c_uint32.from_address(record).value
            if code in (0xc0000005,0xc0000094,0xc0000095,0xc0000409):
                registers={name:hex(C.c_uint64.from_address(context+offset).value) for name,offset in
                    [('rax',0x78),('rcx',0x80),('rdx',0x88),('rbx',0x90),('rsp',0x98),
                     ('rsi',0xa8),('rdi',0xb0),('r8',0xb8),('r9',0xc0),('r10',0xc8),('r11',0xd0),
                     ('r12',0xd8),('r13',0xe0),('r14',0xe8),('r15',0xf0),('rip',0xf8)]}
                rip=int(registers['rip'],16);rsp=int(registers['rsp'],16)
                stack=struct.unpack('<32Q',C.string_at(rsp,256))
                emit('native_exception',code=hex(code),rip_rva=hex(rip-self.base),registers=registers,
                     nearby_return_rvas=[hex(p-self.base) for p in stack if self.base<=p<self.base+self.pe.OPTIONAL_HEADER.SizeOfImage])
        except Exception:pass
        return 0

    def call(self,rva,result,args,*values):
        emit('calling',rva=hex(rva))
        value=C.WINFUNCTYPE(result,*args)(self.base+rva)(*values)
        emit('returned',rva=hex(rva),value=value)
        return value

    def initialize_image_locks(self):
        # The two inspected CRT initializers construct these exact arrays.
        # Call their constructors directly, without unrelated application globals.
        for index in range(16):
            self.call(0xe94c00,C.c_void_p,[C.c_void_p],self.base+0x1e867f0+index*0xb8)
        for index in range(3):
            self.call(0x1000,C.c_void_p,[C.c_void_p],self.base+0x1e866a0+index*0x70)
        emit('image_lock_arrays_initialized',waiters=16,registry_heads=3)

    def initialize_full_crt(self):
        # Keep all application constructors inside this disposable process. The
        # GUI entry point is never called, and foreground-creating imports fail.
        blocked={'CreateWindowExA','CreateWindowExW','ShowWindow','ShowWindowAsync',
                 'SetForegroundWindow','SetActiveWindow','SetFocus','BringWindowToTop',
                 'MessageBoxA','MessageBoxW','MessageBoxExA','MessageBoxExW',
                 'MessageBoxIndirectA','MessageBoxIndirectW','ShellExecuteA','ShellExecuteW',
                 'ShellExecuteExA','ShellExecuteExW','DialogBoxParamA','DialogBoxParamW',
                 'DialogBoxIndirectParamA','DialogBoxIndirectParamW'}
        for name in sorted(blocked & self.import_slots.keys()):
            def deny(name=name):
                self.blocked_ui.append(name);emit('prevented_foreground_call',api=name);return 0
            callback=C.WINFUNCTYPE(C.c_int64)(deny);self.callbacks.append(callback)
            address=self.import_slots[name];old=C.c_uint32();ignored=C.c_uint32()
            self.k.VirtualProtect(address,8,4,C.byref(old))
            C.c_uint64.from_address(address).value=C.cast(callback,C.c_void_p).value
            self.k.VirtualProtect(address,8,old.value,C.byref(ignored))
        self.call(0x13a2210,None,[])
        assert self.call(0x13a1f20,C.c_int,[])>=0
        command=self.call(0x13a1f18,C.c_void_p,[])
        C.c_uint64.from_address(self.base+0x1f40ce8).value=command
        environment=self.call(0x13a1e8c,C.c_void_p,[])
        C.c_uint64.from_address(self.base+0x1f3ceb0).value=environment
        assert self.call(0x13a1d9c,C.c_int,[])>=0
        assert self.call(0x13a1acc,C.c_int,[])>=0
        status=self.call(0x1393288,C.c_int,[C.c_int],1)
        emit('crt_constructors_complete',status=status,gui_entry_called=False,blocked_ui_calls=self.blocked_ui)
        assert status==0
