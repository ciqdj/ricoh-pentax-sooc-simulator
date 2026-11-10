"""Separate immutable bundled resources from writable per-user files."""
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, sys, tempfile, zipfile

VERSION = '3.1'
ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
CONFIG = Path(os.environ.get('APPDATA', str(Path.home() / 'AppData/Roaming'))) / 'DCU5DataTransfer'
LOCAL = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData/Local'))) / 'DCU5DataTransfer'
DEFAULT_OUTPUT = Path(os.environ.get('USERPROFILE', str(Path.home()))) / 'Pictures/DCU5DataTransfer'

def writable_dir(name):
    target = LOCAL / name
    target.mkdir(parents=True, exist_ok=True)
    return target

def report_dir():
    if '--report-dir' in sys.argv:
        target = Path(sys.argv[sys.argv.index('--report-dir') + 1]).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target
    return writable_dir('logs')

def _name(name):
    path = PurePosixPath(str(name).replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or ':' in str(path):
        raise ValueError('Invalid resource path')
    return path.as_posix()

def _index():
    return json.loads((ROOT / 'resource-index.json').read_text(encoding='utf-8'))

def resource_names(prefix):
    prefix = _name(prefix).rstrip('/') + '/'
    if (ROOT / 'assets.zip').is_file():
        return sorted(n for n in _index()['files'] if n.startswith(prefix))
    return sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / prefix).rglob('*') if p.is_file())

def resource_bytes(name):
    name = _name(name)
    if (ROOT / 'assets.zip').is_file():
        index = _index()
        with zipfile.ZipFile(ROOT / 'assets.zip') as bundle:
            data = bundle.read(name)
        if hashlib.sha256(data).hexdigest() != index['files'][name]:
            raise ValueError('Bundled resource checksum mismatch: ' + name)
        return data
    return (ROOT / name).read_bytes()

def resource_file(name):
    """Extract only the requested large LUT/profile, with a versioned cache."""
    name = _name(name)
    if not (ROOT / 'assets.zip').is_file():
        path = ROOT / name
        if not path.is_file(): raise FileNotFoundError(name)
        return path
    index = _index()
    dest = writable_dir('cache') / index['revision'] / name
    if dest.is_file():
        with dest.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() == index['files'][name]:
                return dest
    data = resource_bytes(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, suffix='.tmp', delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(data)
    try: os.replace(tmp, dest)
    finally: tmp.unlink(missing_ok=True)
    return dest

def export_resources(folder, prefix='exports'):
    """Materialize included assets without replacing a user's changed files."""
    folder = Path(folder)
    paths = []
    for name in resource_names(prefix):
        dest = folder / name
        data = resource_bytes(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.read_bytes() != data:
                raise ValueError('同名文件已存在，未覆盖：' + str(dest))
        else:
            with dest.open('xb') as handle: handle.write(data)
        paths.append(dest)
    return paths

def ffmpeg_path():
    bundled = ROOT / 'bin/ffmpeg.exe'
    if bundled.is_file(): return str(bundled)
    if not getattr(sys, 'frozen', False):
        found = shutil.which('ffmpeg')
        if found: return found
    raise FileNotFoundError('内置视频组件缺失，请重新取得完整的单文件版 EXE。')
