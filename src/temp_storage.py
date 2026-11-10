"""Writable local temporary storage, budgeted before full-resolution work.

All returned directories belong to the current task. Existing cache directories
and user files are never removed. The budget is conservative disk headroom, not
an assertion that every native working buffer is written to disk.
"""
from pathlib import Path
import os
import shutil
import tempfile
import time

MIB = 1024 * 1024


class TemporaryStorageError(RuntimeError):
    pass


class TemporaryDirectory(tempfile.TemporaryDirectory):
    """Allow a just-exited Windows worker to release its own file handles."""
    def cleanup(self):
        for attempt in range(20):
            try:
                return super().cleanup()
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.1)


def _pixels(width, height):
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError('照片尺寸无效，无法预估临时空间')
    return width * height


def native_worker_required_bytes(width, height, concurrent=1):
    """RGB16 result + one RGB64 scratch allowance for each active worker."""
    concurrent = max(1, int(concurrent))
    return _pixels(width, height) * (6 + 8) * concurrent + 64 * MIB


def native_required_bytes(width, height, concurrent=2, source_bytes=0):
    """Two immutable RGB16 input banks plus bounded concurrent worker space.

    A same-camera PEF copy may also be needed. The existing input original is
    read-only and never copied unless the native mode explicitly needs it.
    """
    return (12 * _pixels(width, height)
            + native_worker_required_bytes(width, height, concurrent)
            + max(0, int(source_bytes)))


def _existing_parent(path):
    current = Path(path).absolute()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _size_text(size):
    if size >= 1024 * MIB:
        return f'{size / (1024 * MIB):.2f} GB'
    return f'{size / MIB:.1f} MB'


def _space_error(required_bytes, free=None):
    available = '' if free is None else f'，目前可用 {_size_text(max(0, free))}'
    return TemporaryStorageError(
        f'照片临时空间不足或缓存位置不可写：本次需要至少 {_size_text(required_bytes)}'
        f'{available}。请释放磁盘空间，或将 PHOTO_CACHE_DIR 设置为有足够空间的本地缓存文件夹后重试。')


def ensure_free_space(path, required_bytes):
    """Recheck the selected volume immediately before a substantial write."""
    required_bytes = max(0, int(required_bytes))
    try:
        free = int(shutil.disk_usage(_existing_parent(path)).free)
    except OSError:
        raise _space_error(required_bytes) from None
    if free < required_bytes:
        raise _space_error(required_bytes, free)
    return free


def _fixed_local_roots():
    if os.name != 'nt':
        return []
    import ctypes
    kernel = ctypes.windll.kernel32
    kernel.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
    kernel.GetDriveTypeW.restype = ctypes.c_uint
    kernel.GetLogicalDrives.restype = ctypes.c_uint
    mask = kernel.GetLogicalDrives()
    return [Path(chr(65 + i) + ':/') for i in range(26)
            if mask & (1 << i) and kernel.GetDriveTypeW(chr(65 + i) + ':\\') == 3]


def _local_directory(path):
    if os.name != 'nt':
        return True
    import ctypes
    # Resolve junctions so a path that looks local cannot hide a network share.
    anchor = str(Path(path).resolve().anchor)
    kernel = ctypes.windll.kernel32
    kernel.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
    kernel.GetDriveTypeW.restype = ctypes.c_uint
    return bool(anchor) and kernel.GetDriveTypeW(anchor) == 3


def candidate_directories(preferred=None):
    """Explicit choice, launcher choice, normal TEMP, then actual fixed disks."""
    try:
        system_temp = tempfile.gettempdir()
    except OSError:
        # Python probes TEMP when first resolving it. A completely full or
        # unwritable system volume must not prevent trying other local disks.
        system_temp = os.environ.get('TEMP') or os.environ.get('TMP')
    candidates = [preferred, os.environ.get('PHOTO_CACHE_DIR'),
                  os.environ.get('PHOTO_WORKBENCH_TEMP'), system_temp]
    candidates.extend(root / 'PhotoWorkbenchTemp' for root in _fixed_local_roots())
    seen = set()
    for value in candidates:
        if not value:
            continue
        try:
            path = Path(value).expanduser().absolute()
            key = os.path.normcase(str(path.resolve()))
        except (OSError,ValueError):
            continue
        if key not in seen:
            seen.add(key)
            yield path


def _probe_writable(folder):
    probe = Path(folder) / '.write-probe'
    try:
        with probe.open('xb') as stream:
            stream.write(b'\0' * 4096)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        probe.unlink(missing_ok=True)


def temporary_directory(required_bytes, *, prefix='PhotoWorkbench-', preferred=None):
    """Choose sufficient writable storage without assuming any drive letter."""
    required_bytes = max(0, int(required_bytes))
    for base in candidate_directories(preferred):
        owned = None
        try:
            if not _local_directory(base):
                continue
            ensure_free_space(base, required_bytes)
            base.mkdir(parents=True, exist_ok=True)
            owned = TemporaryDirectory(prefix=prefix, dir=base)
            _probe_writable(owned.name)
            ensure_free_space(owned.name, required_bytes)
            return owned
        except (OSError, TemporaryStorageError):
            if owned is not None:
                owned.cleanup()
    raise _space_error(required_bytes)


def subprocess_environment(work):
    """Select storage before the child's bootloader and Python imports run."""
    env = os.environ.copy()
    env['TEMP'] = env['TMP'] = str(Path(work).absolute())
    return env
