"""Tell the native launcher when the first workbench frame is ready."""
import os


def signal_ready():
    name=os.environ.pop('PHOTO_WORKBENCH_READY_EVENT','')
    if not name or os.name!='nt':return False
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenEventW.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.OpenEventW.restype=wintypes.HANDLE
    kernel.SetEvent.argtypes=[wintypes.HANDLE];kernel.SetEvent.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    event=kernel.OpenEventW(2,False,name)
    if not event:return False
    try:return bool(kernel.SetEvent(event))
    finally:kernel.CloseHandle(event)


def after_first_frame(root,on_ready=None):
    # The first timer follows the initial layout/map/paint idle work. A launch
    # that is still mapping keeps the splash, not a guessed percentage.
    if not root.winfo_viewable():
        root.after(40,lambda:after_first_frame(root,on_ready));return
    root.update_idletasks()
    signal_ready()
    if on_ready is not None:root.after(0,on_ready)
