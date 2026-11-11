"""User-owned standalone CUBE LUTs, an atomic local library, and RGB16 rendering.

CUBE syntax/order are checked against OpenColorIO's primary implementations:
https://raw.githubusercontent.com/AcademySoftwareFoundation/OpenColorIO/main/src/OpenColorIO/fileformats/FileFormatIridasCube.cpp
https://raw.githubusercontent.com/AcademySoftwareFoundation/OpenColorIO/main/src/OpenColorIO/fileformats/FileFormatResolveCube.cpp

Independent 1D/3D tables are supported. A combined 1D shaper + 3D file is
rejected, never interpreted as an independent table. Output table values may
be outside [0, 1]; clipping occurs only on delivery to the integer RGB image.
The image adapter supplies display-referred sRGB, not a guessed camera Log
encoding. The user must choose a LUT suitable for this input colour space.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import re
import shlex
import tempfile
import threading
import time
import unicodedata
import uuid

import numpy as np

from native_backend import RenderCancelled

MAX_FILE_BYTES = 1024 * 1024 * 1024
MAX_1D_SIZE = 65536
MAX_3D_SIZE = 256
CHUNK_PIXELS = 65536
_HEX32 = re.compile(r'^[0-9a-f]{32}$')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


class CubeError(ValueError):
    """A readable CUBE syntax, integrity, or unsupported-feature error."""


def _cancel(cancelled):
    if cancelled and cancelled():
        raise RenderCancelled()


def _finite_values(tokens, count, line):
    if len(tokens) != count:
        raise CubeError(f'CUBE 第 {line} 行应包含 {count} 个数值。')
    try:
        result = tuple(float(v) for v in tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CubeError(f'CUBE 第 {line} 行的数值无效。') from exc
    if not all(math.isfinite(v) for v in result):
        raise CubeError(f'CUBE 第 {line} 行含有 NaN 或无穷大。')
    return result


@dataclass(frozen=True)
class Cube:
    kind: str
    size: int
    title: str
    domain_min: tuple
    domain_max: tuple
    table: np.ndarray
    sha256: str

    def transform(self, rgb, cancelled=None):
        """Float64 linear/trilinear interpolation; returns unclipped float64."""
        arr = np.asarray(rgb)
        if arr.ndim < 1 or arr.shape[-1] != 3:
            raise ValueError('LUT 输入必须包含 RGB 三通道。')
        if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
            raise ValueError('LUT 输入必须为实数。')
        flat = arr.reshape(-1, 3)
        out = np.empty(flat.shape, dtype=np.float64)
        lo = np.asarray(self.domain_min, dtype=np.float64)
        span = np.asarray(self.domain_max, dtype=np.float64) - lo
        n = self.size
        table = self.table.reshape(n, n, n, 3) if self.kind == '3D' else self.table
        for start in range(0, len(flat), CHUNK_PIXELS):
            _cancel(cancelled)
            value = np.asarray(flat[start:start + CHUNK_PIXELS], dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError('LUT 输入含有 NaN 或无穷大。')
            # Clamp before normalization, also avoiding subtraction overflow for
            # a finite value very far outside a small input domain.
            p = ((np.clip(value, lo, lo + span) - lo) / span) * (n - 1)
            lower = np.floor(p).astype(np.intp)
            upper = np.minimum(lower + 1, n - 1)
            f = p - lower
            if self.kind == '1D':
                channel = np.arange(3)
                result = table[lower, channel] * (1.0 - f) + table[upper, channel] * f
            else:
                # CUBE writes R fastest, then G, then B: table[b, g, r].
                result = np.zeros(value.shape, dtype=np.float64)
                for b in (0, 1):
                    for g in (0, 1):
                        for r in (0, 1):
                            ix = upper[:, 0] if r else lower[:, 0]
                            iy = upper[:, 1] if g else lower[:, 1]
                            iz = upper[:, 2] if b else lower[:, 2]
                            weight = ((f[:, 0] if r else 1.0 - f[:, 0]) *
                                      (f[:, 1] if g else 1.0 - f[:, 1]) *
                                      (f[:, 2] if b else 1.0 - f[:, 2]))
                            result += table[iz, iy, ix] * weight[:, None]
            if not np.isfinite(result).all():
                raise CubeError('LUT 数值过大，无法安全生成图像。')
            out[start:start + len(value)] = result
        _cancel(cancelled)
        return out.reshape(arr.shape)


def parse_cube(path, cancelled=None):
    """Parse a standalone CUBE without guessing unsupported header semantics."""
    path = Path(path)
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise CubeError('无法读取 LUT 文件。') from exc
    if size_bytes <= 0 or size_bytes > MAX_FILE_BYTES:
        raise CubeError('LUT 文件为空或超过 1 GiB 大小限制。')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        prefix = stream.read(4)
        stream.seek(0)
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            _cancel(cancelled)
            digest.update(block)
    encoding = 'utf-16' if prefix.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    headers = {}
    table = None
    count = 0
    kind = None
    dimension = None
    started = False
    recognized = {'TITLE', 'LUT_1D_SIZE', 'LUT_3D_SIZE', 'DOMAIN_MIN', 'DOMAIN_MAX',
                  'LUT_1D_INPUT_RANGE', 'LUT_3D_INPUT_RANGE', 'INPUT_RANGE'}
    try:
        with path.open('r', encoding=encoding, newline=None) as stream:
            for line_number, original in enumerate(stream, 1):
                if line_number % 2048 == 0:
                    _cancel(cancelled)
                if len(original) > 65536:
                    raise CubeError(f'CUBE 第 {line_number} 行过长。')
                line = original.strip()
                if not line or line.startswith('#'):
                    continue
                first = line.split(None, 1)[0]
                if first == 'TITLE':
                    try:
                        tokens = shlex.split(line, comments=True, posix=True)
                    except ValueError as exc:
                        raise CubeError(f'CUBE 第 {line_number} 行的 TITLE 引号不完整。') from exc
                else:
                    tokens = line.split('#', 1)[0].split()
                if not tokens:
                    continue
                directive = tokens[0]
                if directive in recognized:
                    if started:
                        raise CubeError(f'CUBE 第 {line_number} 行：设置必须写在数据之前。')
                    if directive in headers:
                        raise CubeError(f'CUBE 第 {line_number} 行：{directive} 重复。')
                    if directive == 'TITLE':
                        if len(tokens) != 2 or any(ord(c) < 32 for c in tokens[1]):
                            raise CubeError('CUBE TITLE 必须是单个完整标题。')
                        headers[directive] = tokens[1]
                    elif directive.endswith('_SIZE'):
                        if len(tokens) != 2 or not re.fullmatch(r'[0-9]+', tokens[1]):
                            raise CubeError(f'CUBE 第 {line_number} 行的维度无效。')
                        new_kind = '1D' if directive == 'LUT_1D_SIZE' else '3D'
                        if kind is not None:
                            raise CubeError('暂不支持同一 CUBE 中组合 1D shaper 与 3D LUT；请导出独立 1D 或 3D 文件。')
                        dimension = int(tokens[1])
                        limit = MAX_1D_SIZE if new_kind == '1D' else MAX_3D_SIZE
                        if not 2 <= dimension <= limit:
                            raise CubeError(f'{new_kind} LUT 维度必须为 2–{limit}。')
                        kind = new_kind
                        headers[directive] = dimension
                    else:
                        headers[directive] = _finite_values(tokens[1:], 3 if directive.startswith('DOMAIN_') else 2, line_number)
                    continue
                # A misspelled directive must not be accepted as a data row.
                try:
                    values = _finite_values(tokens, 3, line_number)
                except CubeError as exc:
                    if directive and directive[0].isalpha() and directive.lower() not in ('nan', 'inf', 'infinity'):
                        raise CubeError(f'CUBE 第 {line_number} 行包含不支持的设置：{directive}。') from exc
                    raise
                if kind is None:
                    raise CubeError('CUBE 数据之前缺少 LUT_1D_SIZE 或 LUT_3D_SIZE。')
                if table is None:
                    expected = dimension if kind == '1D' else dimension ** 3
                    table = np.empty((expected, 3), dtype=np.float64)
                started = True
                if count >= len(table):
                    raise CubeError('CUBE 数据行数超过声明维度。')
                table[count] = values
                count += 1
    except UnicodeError as exc:
        raise CubeError('LUT 文本编码无效；请使用 UTF-8、ASCII 或带 BOM 的 UTF-16 文件。') from exc
    _cancel(cancelled)
    if table is None or count != len(table):
        expected = dimension if kind == '1D' else (dimension ** 3 if dimension else 0)
        raise CubeError(f'CUBE 数据行数不完整：需要 {expected} 行，实际 {count} 行。')
    inappropriate = 'LUT_3D_INPUT_RANGE' if kind == '1D' else 'LUT_1D_INPUT_RANGE'
    if inappropriate in headers:
        raise CubeError('CUBE 的 INPUT_RANGE 与 LUT 类型不一致。')
    ranges = [headers[k] for k in ('INPUT_RANGE', f'LUT_{kind}_INPUT_RANGE') if k in headers]
    if len(ranges) == 2 and ranges[0] != ranges[1]:
        raise CubeError('CUBE 中的 INPUT_RANGE 设置互相冲突。')
    lo = headers.get('DOMAIN_MIN', (0., 0., 0.))
    hi = headers.get('DOMAIN_MAX', (1., 1., 1.))
    if ranges:
        a, b = ranges[0]
        if 'DOMAIN_MIN' in headers and lo != (a, a, a):
            raise CubeError('CUBE 的 DOMAIN_MIN 与 INPUT_RANGE 冲突。')
        if 'DOMAIN_MAX' in headers and hi != (b, b, b):
            raise CubeError('CUBE 的 DOMAIN_MAX 与 INPUT_RANGE 冲突。')
        lo, hi = (a, a, a), (b, b, b)
    spans = [b - a for a, b in zip(lo, hi)]
    if not all(math.isfinite(v) and v > 0 for v in spans):
        raise CubeError('CUBE 输入范围上限必须大于下限，且跨度必须为有限数。')
    table.setflags(write=False)
    return Cube(kind, dimension, headers.get('TITLE', ''), tuple(lo), tuple(hi), table, digest.hexdigest())


def _name(value):
    if not isinstance(value, str):
        raise ValueError('名称必须是文字。')
    value = unicodedata.normalize('NFC', value.strip())
    if not value or len(value) > 120 or any(unicodedata.category(c).startswith('C') for c in value):
        raise ValueError('名称须为 1–120 个字符，不能含控制字符。')
    return value


def _new_state():
    return {'version': 1, 'brands': [], 'categories': [], 'luts': []}


def _atomic_json(path, data):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.025 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


class Library:
    """Library records and original CUBE files. Display names are never paths."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.index_path = self.root / 'index.json'
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._cubes = OrderedDict()
        self._index_marker = None
        self._index_cache = None

    @contextmanager
    def _transaction(self):
        # A slow import may validate a large existing object. Keep the writer
        # mutex separate from the short metadata/cache mutex used by the UI.
        with self._write_lock:
            self.root.mkdir(parents=True, exist_ok=True)
            lock_path = self.root / '.library.lock'
            with lock_path.open('a+b') as stream:
                if stream.tell() == 0:
                    stream.write(b'0')
                    stream.flush()
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    deadline = time.monotonic() + 15
                    while True:
                        try:
                            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                            break
                        except OSError:
                            if time.monotonic() >= deadline:
                                raise RuntimeError('滤镜库正在由另一任务写入，请稍后重试。')
                            time.sleep(.05)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    stream.seek(0)
                    if os.name == 'nt':
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read(self, mutable=True):
        with self._lock:
            return self._read_locked(mutable)

    def _read_locked(self, mutable=True):
        if not self.index_path.exists():
            return _new_state()
        try:
            stat = self.index_path.stat()
            marker = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if marker == self._index_marker and self._index_cache is not None:
                return copy.deepcopy(self._index_cache) if mutable else self._index_cache
            if stat.st_size > 16 * 1024 * 1024:
                raise ValueError('index too large')
            state = json.loads(self.index_path.read_text(encoding='utf-8'))
            if set(state) != {'version', 'brands', 'categories', 'luts'} or state['version'] != 1:
                raise ValueError('schema')
            brands, categories, keys = set(), set(), set()
            for item in state['brands']:
                if set(item) != {'id', 'name'} or not _HEX32.fullmatch(item['id']) or item['id'] in brands:
                    raise ValueError('brand')
                _name(item['name'])
                brands.add(item['id'])
            for item in state['categories']:
                if set(item) != {'id', 'brand_id', 'name'} or not _HEX32.fullmatch(item['id']) or item['id'] in categories or item['brand_id'] not in brands:
                    raise ValueError('category')
                _name(item['name'])
                categories.add(item['id'])
            category_brand = {v['id']: v['brand_id'] for v in state['categories']}
            for item in state['luts']:
                if not _HEX32.fullmatch(item['id']) or item['key'] != 'custom:' + item['id'] or item['key'] in keys:
                    raise ValueError('key')
                if not _SHA256.fullmatch(item['sha256']) or item['brand_id'] not in brands or category_brand.get(item['category_id']) != item['brand_id']:
                    raise ValueError('reference')
                if item['kind'] not in ('1D', '3D') or type(item['size']) is not int or not 2 <= item['size'] <= (MAX_1D_SIZE if item['kind'] == '1D' else MAX_3D_SIZE):
                    raise ValueError('table')
                if set(item) != {'id', 'key', 'name', 'brand_id', 'category_id', 'title', 'kind', 'size', 'sha256', 'domain_min', 'domain_max'}:
                    raise ValueError('fields')
                _name(item['name'])
                if not isinstance(item['title'], str) or any(ord(c) < 32 for c in item['title']):
                    raise ValueError('title')
                if len(item['domain_min']) != 3 or len(item['domain_max']) != 3:
                    raise ValueError('domain')
                for a, b in zip(item['domain_min'], item['domain_max']):
                    if not all(math.isfinite(v) for v in (a, b, b - a)) or b <= a:
                        raise ValueError('domain')
                keys.add(item['key'])
            self._index_marker, self._index_cache = marker, state
            return copy.deepcopy(state) if mutable else state
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError('自定义滤镜库索引损坏；已保留原数据，没有重置或覆盖。') from exc

    def brands(self):
        with self._lock:
            return {v['id']: v['name'] for v in self._read(False)['brands']}

    def categories(self, brand_id):
        with self._lock:
            return {v['id']: v['name'] for v in self._read(False)['categories'] if v['brand_id'] == brand_id}

    def catalog(self, brand_id=None, category_id=None):
        with self._lock:
            return {v['key']: copy.deepcopy(v) for v in self._read(False)['luts']
                    if (brand_id is None or v['brand_id'] == brand_id) and
                       (category_id is None or v['category_id'] == category_id)}

    def records(self, brand_id=None):
        return list(self.catalog(brand_id).values())

    def list_brands(self):
        return [{'id': key, 'name': value} for key, value in self.brands().items()]

    def list_categories(self, brand_id):
        return [{'id': key, 'brand_id': brand_id, 'name': value} for key, value in self.categories(brand_id).items()]

    def list_luts(self, brand_id=None, category_id=None):
        return list(self.catalog(brand_id, category_id).values())

    def create_brand(self, name):
        name = _name(name)
        with self._transaction():
            state = self._read()
            for item in state['brands']:
                if item['name'].casefold() == name.casefold():
                    return copy.deepcopy(item)
            item = {'id': uuid.uuid4().hex, 'name': name}
            state['brands'].append(item)
            state['categories'].append({'id': uuid.uuid4().hex, 'brand_id': item['id'], 'name': '未分类'})
            _atomic_json(self.index_path, state)
            return copy.deepcopy(item)

    def create_category(self, brand_id, name):
        name = _name(name)
        with self._transaction():
            state = self._read()
            if brand_id not in {v['id'] for v in state['brands']}:
                raise ValueError('所选品牌已不存在。')
            for item in state['categories']:
                if item['brand_id'] == brand_id and item['name'].casefold() == name.casefold():
                    return copy.deepcopy(item)
            item = {'id': uuid.uuid4().hex, 'brand_id': brand_id, 'name': name}
            state['categories'].append(item)
            _atomic_json(self.index_path, state)
            return copy.deepcopy(item)

    add_brand = create_brand
    add_group = create_category

    def lookup(self, key):
        with self._lock:
            item = next((v for v in self._read(False)['luts'] if v['key'] == key), None)
            item = copy.deepcopy(item) if item is not None else None
        if item is None:
            raise ValueError('所选自定义滤镜已不存在，请重新选择。')
        return item

    def _object_path(self, digest):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CubeError('滤镜库数据对象的标识无效。')
        path = (self.root / 'objects' / (digest + '.cube')).resolve()
        if not path.is_relative_to(self.root):
            raise CubeError('滤镜库数据对象位于库目录之外；没有读取或写入。')
        return path

    def _cube_for(self, item, cancelled=None):
        path = self._object_path(item['sha256'])
        try:
            stat = path.stat()
        except OSError as exc:
            raise CubeError('滤镜库中的 CUBE 文件缺失，请重新导入。') from exc
        marker = (item['sha256'], stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with self._lock:
            cube = self._cubes.get(marker)
            if cube is not None:
                self._cubes.move_to_end(marker)
                return path, cube
        cube = parse_cube(path, cancelled)
        if cube.sha256 != item['sha256'] or cube.kind != item['kind'] or cube.size != item['size'] or list(cube.domain_min) != item['domain_min'] or list(cube.domain_max) != item['domain_max']:
            raise CubeError('滤镜库中的 CUBE 与索引不一致；没有应用该文件。')
        with self._lock:
            self._cubes[marker] = cube
            while len(self._cubes) > 2 or (len(self._cubes) > 1 and sum(c.table.nbytes for c in self._cubes.values()) > 128 * 1024 * 1024):
                self._cubes.popitem(last=False)
        return path, cube

    def get_cube_path(self, key):
        path, _ = self._cube_for(self.lookup(key))
        return path

    def get_cube(self, key, cancelled=None):
        _cancel(cancelled)
        return self._cube_for(self.lookup(key), cancelled)[1]

    def import_cube(self, path, brand_id, category_id=None, name=None, cancelled=None):
        return self._import([path], brand_id, category_id, [name], cancelled, None)[0]

    def import_files(self, paths, brand_id, category_id=None, cancelled=None, progress=None):
        paths = list(paths)
        return self._import(paths, brand_id, category_id, [None] * len(paths), cancelled, progress)

    def _import(self, paths, brand_id, category_id, names, cancelled, progress):
        if not paths:
            return []
        self.root.mkdir(parents=True, exist_ok=True)
        staged = []
        # TemporaryDirectory is rooted in the library, never the system drive.
        with tempfile.TemporaryDirectory(prefix='.import-', dir=self.root) as staging:
            for i, (source, display_name) in enumerate(zip(paths, names)):
                _cancel(cancelled)
                source = Path(source)
                if source.suffix.lower() != '.cube':
                    raise CubeError('请选择 .cube 格式的 LUT。')
                target = Path(staging) / f'{i}.cube'
                count = 0
                with source.open('rb') as inp, target.open('xb') as out:
                    while True:
                        block = inp.read(1024 * 1024)
                        if not block:
                            break
                        _cancel(cancelled)
                        count += len(block)
                        if count > MAX_FILE_BYTES:
                            raise CubeError('LUT 文件超过 1 GiB 大小限制。')
                        out.write(block)
                cube = parse_cube(target, cancelled)
                label = _name(display_name) if display_name is not None else _name((cube.title.strip() or source.stem)[:120])
                staged.append((target, cube, label))
                if progress:
                    progress(f'已检查 LUT {i + 1}/{len(paths)}')
            with self._transaction():
                _cancel(cancelled)
                state = self._read()
                if brand_id not in {v['id'] for v in state['brands']}:
                    raise ValueError('所选品牌已不存在。')
                options = [v for v in state['categories'] if v['brand_id'] == brand_id]
                selected_category = category_id
                if selected_category is None:
                    selected_category = next((v['id'] for v in options if v['name'] == '未分类'), options[0]['id'] if options else None)
                if selected_category not in {v['id'] for v in options}:
                    raise ValueError('所选分类不属于当前品牌。')
                result, created = [], []
                objects = self.root / 'objects'
                try:
                    changed = False
                    for staged_path, cube, label in staged:
                        _cancel(cancelled)
                        matching = [v for v in state['luts'] if v['brand_id'] == brand_id and v['category_id'] == selected_category]
                        repeated = next((v for v in matching if v['sha256'] == cube.sha256), None)
                        if repeated is not None:
                            # An existing record is never renamed by re-import.
                            self._cube_for(repeated, cancelled)
                            result.append(copy.deepcopy(repeated))
                            continue
                        occupied = {v['name'].casefold() for v in matching}
                        base_label, number = label, 2
                        while label.casefold() in occupied:
                            suffix = f' ({number})'
                            label = base_label[:120 - len(suffix)] + suffix
                            number += 1
                        internal = self._object_path(cube.sha256)
                        objects.mkdir(exist_ok=True)
                        if internal.exists():
                            # Check an existing immutable object before linking a
                            # new record; do not overwrite a damaged old object.
                            if parse_cube(internal, cancelled).sha256 != cube.sha256:
                                raise CubeError('库中已有同名数据对象损坏；已保留原文件。')
                        else:
                            os.replace(staged_path, internal)
                            created.append(internal)
                        ident = uuid.uuid4().hex
                        item = {'id': ident, 'key': 'custom:' + ident, 'name': label,
                                'brand_id': brand_id, 'category_id': selected_category,
                                'title': cube.title, 'kind': cube.kind, 'size': cube.size,
                                'sha256': cube.sha256, 'domain_min': list(cube.domain_min),
                                'domain_max': list(cube.domain_max)}
                        state['luts'].append(item)
                        result.append(copy.deepcopy(item))
                        changed = True
                    _cancel(cancelled)
                    if changed:
                        _atomic_json(self.index_path, state)
                    return result
                except BaseException:
                    for path in created:
                        path.unlink(missing_ok=True)
                    raise


_libraries = {}
_libraries_lock = threading.Lock()


def get_library(root=None):
    include_bundled=root is None
    if root is None:
        import app_paths
        root = Path(app_paths.CONFIG) / 'custom_luts'
    root = Path(root).resolve()
    with _libraries_lock:
        key=(root,include_bundled)
        if key not in _libraries:
            library=Library(root)
            if include_bundled:
                from builtin_luts import with_bundled
                library=with_bundled(library)
            _libraries[key]=library
        return _libraries[key]


def _parameters(amount, exposure, controls=None):
    try:
        amount, exposure = float(amount), float(exposure)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('LUT 浓淡或曝光设置无效。') from exc
    if not math.isfinite(amount) or not 0 <= amount <= 2 or not math.isfinite(exposure) or not -5 <= exposure <= 5:
        raise ValueError('LUT 浓淡范围为 0–200%，曝光范围为 -5–+5 EV。')
    if controls:
        raise ValueError('自定义 CUBE 不支持原厂滤镜专属参数。')
    return amount, exposure


def apply_cube(rgb, cube, amount=1, exposure=0, cancelled=None, progress=None):
    """RGB8/16 -> RGB16. EV is linear-light, before LUT and amount mixing."""
    amount, exposure = _parameters(amount, exposure)
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype not in (np.uint8, np.uint16) or not arr.shape[0] or not arr.shape[1]:
        raise ValueError('照片必须是非空 RGB 8 位或 16 位图像。')
    _cancel(cancelled)
    if amount == 0 and exposure == 0:
        return arr if arr.dtype == np.uint16 else arr.astype(np.uint16) * 257
    maximum = float(np.iinfo(arr.dtype).max)
    flat = arr.reshape(-1, 3)
    result = np.empty(flat.shape, dtype=np.uint16)
    for start in range(0, len(flat), CHUNK_PIXELS):
        _cancel(cancelled)
        base = flat[start:start + CHUNK_PIXELS].astype(np.float64) / maximum
        if exposure:
            linear = np.where(base <= .04045, base / 12.92, ((base + .055) / 1.055) ** 2.4)
            linear *= 2.0 ** exposure
            # Keep values above one for a LUT whose declared domain accepts them.
            base = np.where(linear <= .0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - .055)
        if amount:
            clean = cube.transform(base, cancelled)
            composed = clean if amount == 1 else base * (1.0 - amount) + clean * amount
        else:
            composed = base
        if not np.isfinite(composed).all():
            raise CubeError('LUT 的输出或浓淡运算超出可处理数值范围。')
        result[start:start + len(base)] = np.floor(np.clip(composed, 0, 1) * 65535.0 + .5).astype(np.uint16)
        if progress:
            progress(f'应用自定义 LUT {min(100, round(100 * (start + len(base)) / len(flat)))}%')
    _cancel(cancelled)
    return result.reshape(arr.shape)


class CustomSession:
    def __init__(self, source, library=None):
        self.source = Path(source)
        self.library = library if library is not None else get_library()
        self.base = None
        self.alpha = None
        self.size = None
        self.decode_count = 0
        self.lock = threading.RLock()

    def prepare(self, cancelled=None, progress=None):
        with self.lock:
            _cancel(cancelled)
            if self.base is None:
                from photo_formats import read_photo
                from color_math import srgb_encode
                if progress:
                    progress('读取自定义滤镜的完整输入照片')
                photo = read_photo(self.source)
                _cancel(cancelled)
                if photo.rgb.dtype not in (np.uint8, np.uint16):
                    raise ValueError('照片解码器返回了不支持的精度。')
                if photo.linear:
                    if photo.rgb.dtype != np.uint16:
                        raise ValueError('线性照片必须以 16 位精度解码。')
                    encoding = np.floor(np.clip(srgb_encode(np.arange(65536, dtype=np.float64) / 65535), 0, 1) * 65535 + .5).astype(np.uint16)
                    base = np.empty_like(photo.rgb)
                    for row in range(0, len(base), 128):
                        _cancel(cancelled)
                        base[row:row + 128] = encoding[photo.rgb[row:row + 128]]
                else:
                    base = photo.rgb.astype(np.uint16) * 257 if photo.rgb.dtype == np.uint8 else photo.rgb.copy()
                alpha = None
                if photo.alpha is not None:
                    alpha = photo.alpha.astype(np.float64) / np.iinfo(photo.alpha.dtype).max
                    alpha.setflags(write=False)
                _cancel(cancelled)
                base.setflags(write=False)
                self.size = (base.shape[1], base.shape[0])
                self.base, self.alpha = base, alpha
                self.decode_count += 1
            return self.base, self.alpha

    def _report(self, key, amount, exposure, preview_shape=None):
        item = self.library.lookup(key)
        return {'look': key, 'brand': 'custom', 'custom_brand_id': item['brand_id'],
                'custom_category_id': item['category_id'], 'custom_lut_name': item['name'],
                'input_route': 'custom_cube_rgb16', 'native_width': self.size[0], 'native_height': self.size[1],
                'cube_sha256': item['sha256'], 'cube_kind': item['kind'], 'cube_size': item['size'],
                'cube_domain_min': item['domain_min'], 'cube_domain_max': item['domain_max'],
                'lut_interpolation': 'linear' if item['kind'] == '1D' else 'trilinear',
                'input_colour_space': 'sRGB', 'lut_input_assumption': 'display-referred sRGB; no implicit camera Log conversion',
                'output_range': 'clipped to 0..1 only when forming RGB16',
                'exposure_ev': exposure, 'user_amount_percent': amount * 100,
                'preview_only': preview_shape is not None,
                'preview_width': preview_shape[1] if preview_shape else self.size[0],
                'preview_height': preview_shape[0] if preview_shape else self.size[1],
                'personal_metadata_copied': False, 'original_dcu_pipeline': False}

    def baseline(self, cancelled=None, progress=None):
        base, alpha = self.prepare(cancelled, progress)
        return base, alpha, {'brand': 'custom', 'native_width': self.size[0], 'native_height': self.size[1], 'personal_metadata_copied': False}

    def render(self, key, amount=1, exposure=0, cancelled=None, progress=None, controls=None):
        amount, exposure = _parameters(amount, exposure, controls)
        base, alpha = self.prepare(cancelled, progress)
        cube = self.library.get_cube(key, cancelled)
        rgb = apply_cube(base, cube, amount, exposure, cancelled, progress)
        return rgb, alpha, self._report(key, amount, exposure)

    def render_preview(self, key, amount=1, exposure=0, cancelled=None, progress=None, maxsize=(1200, 900), controls=None):
        amount, exposure = _parameters(amount, exposure, controls)
        base, alpha = self.prepare(cancelled, progress)
        if len(maxsize) != 2 or any(type(v) is not int or v <= 0 for v in maxsize):
            raise ValueError('预览尺寸必须是两个正整数。')
        width, height = self.size
        ratio = min(1., maxsize[0] / width, maxsize[1] / height)
        out_w, out_h = max(1, round(width * ratio)), max(1, round(height * ratio))
        if (out_w, out_h) != (width, height):
            xs = np.minimum(width - 1, np.floor((np.arange(out_w) + .5) * width / out_w).astype(np.intp))
            ys = np.minimum(height - 1, np.floor((np.arange(out_h) + .5) * height / out_h).astype(np.intp))
            base = np.ascontiguousarray(base[ys[:, None], xs[None, :]])
            if alpha is not None:
                alpha = np.ascontiguousarray(alpha[ys[:, None], xs[None, :]])
        cube = self.library.get_cube(key, cancelled)
        rgb = apply_cube(base, cube, amount, exposure, cancelled, progress)
        return base, rgb, alpha, self._report(key, amount, exposure, base.shape)

    def close(self):
        with self.lock:
            self.base = self.alpha = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def save(source, destination, key, exposure=0, jpeg_quality=98, bit_depth=16,
         controls=None, amount=1, progress=None, cancelled=None,
         post_balance=(0, 0), post_settings=None):
    """Standard photo backend API, full resolution and stripped personal metadata."""
    from photo_formats import RASTER_EXT, resolve_format, capture_for_output, save_raster
    from postprocess import process as post_process, annotate

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    sidecar = destination.with_name(destination.name + '.json')
    if source == destination or destination.exists() or sidecar.exists():
        raise ValueError('输出必须是新文件，不覆盖原片。')
    if destination.suffix.lower() not in RASTER_EXT:
        raise ValueError('不支持的输出格式。')
    if bit_depth not in (8, 16) or type(jpeg_quality) is not int or not 80 <= jpeg_quality <= 100:
        raise ValueError('导出设置无效。')
    _parameters(amount, exposure, controls)
    with CustomSession(source) as session:
        rgb, alpha, report = session.render(key, amount, exposure, cancelled, progress, controls)
        rgb = post_process(rgb, post_balance, post_settings, cancelled, progress, scale=1.0, alpha=alpha)
        annotate(report, post_balance, post_settings)
        _cancel(cancelled)
        fmt = resolve_format(destination)
        bits = bit_depth if fmt in ('png', 'tiff') else 8
        keep_alpha = alpha is not None and fmt in ('png', 'tiff', 'webp')
        if alpha is not None and not keep_alpha:
            rgb = np.floor(rgb * alpha[..., None] + 65535 * (1 - alpha[..., None]) + .5).astype(np.uint16)
        if bits == 8:
            rgb = ((rgb.astype(np.uint32) + 128) // 257).astype(np.uint8)
        if keep_alpha:
            channel = np.floor(np.clip(alpha, 0, 1) * (65535 if bits == 16 else 255) + .5).astype(rgb.dtype)
            rgb = np.concatenate([rgb, channel[..., None]], axis=2)
        exif,capture=capture_for_output(source,rgb.shape)
        destination.parent.mkdir(parents=True, exist_ok=True)
        created = sidecar_created = False
        try:
            _cancel(cancelled)
            save_raster(destination,rgb,fmt,jpeg_quality,keep_alpha,exif)
            created=True
            _cancel(cancelled)
            report.update(output_format=fmt, output_bits=bits, alpha_retained=keep_alpha,
                          lossless_encoding=fmt != 'jpeg',capture_metadata=capture)
            with sidecar.open('x', encoding='utf-8') as out:
                sidecar_created = True
                json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
            _cancel(cancelled)
            return report
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            if sidecar_created:
                sidecar.unlink(missing_ok=True)
            raise
