"""Bounded style-card rendering, independent of full-photo/AI generations.

Cards use a small RGB16 copy of the original input and the existing renderers.
They are visual selectors, never a source for the main preview or an export.
Only plain arrays/PIL images live here; no Tk objects enter the worker.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import json, threading, weakref, time
import numpy as np
from PIL import Image
from native_backend import RenderCancelled
from photo_denoise import identity_for


@dataclass(frozen=True)
class ThumbnailSpec:
    brand: str
    look: str
    exposure: float = 0.
    controls: tuple = ()
    amount: float = 1.
    balance: tuple = (0., 0.)
    post: str = '{}'
    base: str = '{}'


@dataclass(frozen=True)
class ThumbnailKey:
    photo: object
    spec: ThumbnailSpec


def spec_for(brand, look, exposure=0., controls=None, amount=1., balance=(0., 0.), post=None, base=None):
    from native_controls import NR_CONTROLS
    from postprocess import validate
    settings = validate(post)
    # Photo-wide noise settings must not invalidate a style selector.
    settings = {name: settings[name] for name in ('basic', 'mixer', 'tone')}
    controls = tuple(sorted((name, value) for name, value in (controls or {}).items()
                            if name not in NR_CONTROLS))
    return ThumbnailSpec(brand, look, round(float(exposure), 2), controls,
                         round(float(amount), 2), tuple(balance),
                         json.dumps(settings, sort_keys=True, separators=(',', ':')),
                         json.dumps(base or {}, sort_keys=True, separators=(',', ':')))


class _SmallInput:
    def __init__(self, path, temp, size, source_size):
        self.path = path; self.size = size; self.source_size = source_size
        self._cleanup = weakref.finalize(self, temp.cleanup)


def _lower_priority():
    # Only this one background thread; do not change the application's mode.
    try:
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentThread.restype = ctypes.c_void_p
        kernel.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        kernel.SetThreadPriority(kernel.GetCurrentThread(), -1)
    except (AttributeError, OSError):
        pass


def _prepare(photo_identity, cancelled, prepared=None):
    import tifffile
    from photo_formats import read_photo, RAW_EXT, ICC
    from native_backend import _SRGB16
    from temp_storage import temporary_directory
    if cancelled():raise RenderCancelled()
    if identity_for(photo_identity.path) != photo_identity:raise RenderCancelled()
    if prepared is not None and prepared.identity == photo_identity:
        rgb = prepared.rgb; alpha = prepared.alpha; linear = False
    else:
        # At most once for this photo cache, not once per card. Half RAW is
        # sufficient for a 320px selector and does not affect main decoding.
        photo = read_photo(photo_identity.path,
                           half_size=Path(photo_identity.path).suffix.lower() in RAW_EXT)
        rgb = photo.rgb; alpha = photo.alpha; linear = photo.linear
    if cancelled():raise RenderCancelled()
    h, w = rgb.shape[:2]; ratio = min(1., 320 / w, 240 / h)
    size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    # Use source pixel centres without first quantizing to a JPEG/RGB8 image.
    # Encoding is pointwise and therefore only needed for the sampled pixels.
    if ratio < 1:
        rows = np.minimum(h - 1, ((np.arange(size[1]) + .5) * h / size[1]).astype(np.intp))
        cols = np.minimum(w - 1, ((np.arange(size[0]) + .5) * w / size[0]).astype(np.intp))
        rgb = rgb[rows[:, None], cols[None, :]]
        if alpha is not None:alpha = alpha[rows[:, None], cols[None, :]]
    if rgb.dtype == np.uint8:rgb = rgb.astype(np.uint16) * 257
    if linear:rgb = _SRGB16[rgb]
    if alpha is not None:
        if alpha.dtype == np.uint8:alpha = alpha.astype(np.uint16) * 257
        rgb = np.dstack([rgb, alpha])
    if cancelled():raise RenderCancelled()
    if identity_for(photo_identity.path) != photo_identity:raise RenderCancelled()
    temp = temporary_directory(2 * 1024 * 1024, prefix='PresetCards-')
    path = Path(temp.name) / 'style-input.tif'
    try:
        tifffile.imwrite(path, rgb, photometric='rgb', metadata=None, software=None,
                         iccprofile=ICC, extrasamples='unassalpha' if alpha is not None else None)
        return _SmallInput(path, temp, size, (w, h))
    except BaseException:
        temp.cleanup(); raise


def _render(small, spec, cancelled, library):
    from filter_catalog import PENTAX, RICOH, CUSTOM, CREATIVE_LOOKS
    from native_app import preview
    controls = dict(spec.controls)
    if spec.brand == PENTAX and spec.look not in CREATIVE_LOOKS:
        from native_backend import RenderSession, mix_amount
        with RenderSession(small.path) as session:
            base, alpha = session.baseline(spec.look, cancelled=cancelled)
            rgb, alpha, report = session.render(spec.look, spec.exposure,
                                                cancelled=cancelled, controls=controls)
            rgb = mix_amount(base, rgb, spec.amount)
    elif spec.brand == PENTAX:
        from legacy_effects import CreativeSession
        rgb, alpha, report = CreativeSession(small.path).render(
            spec.look, spec.amount, spec.exposure, cancelled)
    elif spec.brand == RICOH:
        from gr4_backend import GRSession
        rgb, alpha, report = GRSession(small.path).process(
            spec.look, spec.exposure, controls, spec.amount, cancelled, allow_gpu=False)
        if alpha is not None:alpha = alpha.astype(np.float64) / 65535
    elif spec.brand == CUSTOM:
        from custom_lut import CustomSession
        with CustomSession(small.path, library=library) as session:
            rgb, alpha, report = session.render(spec.look, spec.amount, spec.exposure,
                                                cancelled, controls=controls)
    else:raise ValueError('Unknown thumbnail brand')
    if cancelled():raise RenderCancelled()
    settings = json.loads(spec.post)
    # Apply the same manual operations, using the established CPU reference
    # for XY so a background card never initializes/contends for the GPU.
    from postprocess import process
    from color_balance import transform
    from tone_tools import apply_array
    scale = min(small.size[0] / small.source_size[0], small.size[1] / small.source_size[1])
    rgb = process(rgb, (0, 0), {name: settings.get(name) for name in ('basic', 'mixer')},
                  cancelled, scale=scale, alpha=alpha)
    if any(spec.balance):
        rgb = np.floor(transform(rgb.astype(np.float64) / 65535, *spec.balance) * 65535 + .5).astype(np.uint16)
    rgb = apply_array(rgb, settings.get('tone'), cancelled)
    image = preview(rgb, alpha, size=(112, 76))
    return image, dict(report, style_thumbnail=True, source_size=small.source_size,
                       thumbnail_input_size=small.size)


def _committed_input(small, encoded, cancelled):
    """Manual adjustments only. A selected filter/AI image never enters here."""
    from copy import deepcopy
    import tifffile
    from photo_formats import read_photo, ICC
    from photo_edit import normalize
    from editor_pipeline import apply_manual
    from color_balance import transform
    from tone_tools import apply_array
    from temp_storage import temporary_directory
    state=normalize(json.loads(encoded));photo=read_photo(small.path)
    balance=state['balance'];tone=state['post']['tone']
    # Keep the exact operation order while using the existing CPU XY formula.
    cpu=deepcopy(state);cpu['balance']=[0,0];cpu['post']['tone']={}
    scale=min(small.size[0]/small.source_size[0],small.size[1]/small.source_size[1])
    rgb=apply_manual(photo.rgb,cpu,photo.alpha,scale,cancelled)
    if any(balance):rgb=np.floor(transform(rgb.astype(np.float64)/65535,*balance)*65535+.5).astype(np.uint16)
    rgb=apply_array(rgb,tone,cancelled)
    if cancelled():raise RenderCancelled()
    if photo.alpha is not None:rgb=np.dstack([rgb,photo.alpha])
    temp=temporary_directory(2*1024*1024,prefix='CommittedCards-');path=Path(temp.name)/'committed-base.tif'
    try:
        tifffile.imwrite(path,rgb,photometric='rgb',metadata=None,software=None,iccprofile=ICC,
                         extrasamples='unassalpha' if photo.alpha is not None else None)
        return _SmallInput(path,temp,small.size,small.source_size)
    except BaseException:temp.cleanup();raise


class PresetThumbnailStore:
    """One low-priority task, bounded caches, latest visible requests only."""
    def __init__(self, library=None, max_images=384, max_sources=6):
        self.library = library; self.max_images = max_images; self.max_sources = max_sources
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='StyleCards',
                                        initializer=_lower_priority)
        self._cache = OrderedDict(); self._sources = OrderedDict(); self._bases=OrderedDict(); self._lock = threading.Lock()
        self._preferred = OrderedDict(); self._last = OrderedDict()
        self._view = (); self._photo = None; self._future = None; self._running = None
        self._cancel = threading.Event(); self._closed = False; self._failed = {}
        self._stats = {'decode_count': 0, 'prepared_reuse': 0, 'base_count':0,'render_count': 0,
                       'cache_hits': 0, 'cancelled': 0, 'failed': 0}

    def choose(self, photo, card, spec, replace=False):
        slot = (photo, spec.brand, card)
        key = ThumbnailKey(photo, spec)
        if not replace:key = self._preferred.get(slot, key)
        if key in self._cache:
            self._stats['cache_hits'] += 1; self._cache.move_to_end(key)
        self._preferred[slot] = key; self._preferred.move_to_end(slot)
        while len(self._preferred) > self.max_images:self._preferred.popitem(last=False)
        return key

    def image(self, photo, card, key):
        slot = (photo, key.spec.brand, card)
        cached_key = key if key in self._cache else self._last.get(slot)
        cached = self._cache.get(cached_key)
        if cached is None:return None
        self._cache.move_to_end(cached_key)
        return cached[0]

    def remember(self, photo, card, key):
        if key not in self._cache:return
        slot = (photo, key.spec.brand, card)
        self._last[slot] = key; self._last.move_to_end(slot)
        while len(self._last) > self.max_images:self._last.popitem(last=False)

    def set_view(self, photo, keys):
        if photo != self._photo:
            self._photo = photo; self._cancel.set()
        self._view = tuple(dict.fromkeys(keys))

    def _task(self, key, cancel, prepared):
        cancelled = cancel.is_set
        with self._lock:small = self._sources.get(key.photo)
        if small is None:
            small = _prepare(key.photo, cancelled, prepared)
            with self._lock:
                self._sources[key.photo] = small
                self._stats['prepared_reuse' if prepared is not None else 'decode_count'] += 1
                while len(self._sources) > self.max_sources:self._sources.popitem(last=False)
        else:
            with self._lock:self._sources.move_to_end(key.photo)
        if cancelled():raise RenderCancelled()
        if key.spec.base!='{}':
            base_key=(key.photo,key.spec.base)
            with self._lock:committed=self._bases.get(base_key)
            if committed is None:
                committed=_committed_input(small,key.spec.base,cancelled)
                with self._lock:
                    self._bases[base_key]=committed;self._stats['base_count']+=1
                    while len(self._bases)>self.max_sources:self._bases.popitem(last=False)
            else:
                with self._lock:self._bases.move_to_end(base_key)
            small=committed
        result = _render(small, key.spec, cancelled, self.library)
        with self._lock:self._stats['render_count'] += 1
        return result

    def poll(self, paused=False, prepared=None):
        completed = []
        if self._future is not None and self._future.done():
            future, key = self._future, self._running
            self._future = self._running = None
            try:
                value = future.result()
                self._cache[key] = value; self._cache.move_to_end(key)
                while len(self._cache) > self.max_images:self._cache.popitem(last=False)
                completed.append((key, value[0], None))
            except RenderCancelled:
                self._stats['cancelled'] += 1
            except Exception as exc:
                self._failed[key] = str(exc); self._stats['failed'] += 1
                while len(self._failed) > self.max_images:self._failed.pop(next(iter(self._failed)))
                completed.append((key, None, str(exc)))
        if not self._closed and not paused and self._future is None:
            key = next((key for key in self._view if key not in self._cache and key not in self._failed), None)
            if key is not None:
                self._cancel = threading.Event(); self._running = key
                source = prepared if prepared is not None and prepared.identity == key.photo else None
                self._future = self._pool.submit(self._task, key, self._cancel, source)
        return completed

    def retry(self, key):self._failed.pop(key, None)

    def stats(self):
        with self._lock:result = dict(self._stats, sources=len(self._sources))
        return dict(result, images=len(self._cache), running=self._running,
                    queue=sum(key not in self._cache and key not in self._failed for key in self._view),
                    bytes=sum(value[0].width * value[0].height * 3 for value in self._cache.values()))

    def cancel(self):self._closed = True; self._cancel.set(); self._view = ()

    def close(self, wait=True):
        self.cancel(); self._pool.shutdown(wait=wait, cancel_futures=True)
        if wait:
            self._sources.clear();self._bases.clear(); self._cache.clear(); self._preferred.clear(); self._last.clear()


class PresetThumbnailMixin:
    def init_preset_thumbnails(self):
        self.preset_thumbnails = PresetThumbnailStore(self.custom_library)
        self._preset_photo = None; self._preset_requests = {}
        self._preset_context = None; self._preset_card_layout = None
        self._preset_visibility_stamp = None
        self._preset_committed_base=None;self._preset_committed_styles={}

    def set_committed_preset_base(self,base,styles,serial):
        from copy import deepcopy
        self._preset_committed_base=deepcopy(base);self._preset_committed_styles=deepcopy(styles)
        self.bind_preset_cards()

    def preset_photo_selected(self, path):
        self._preset_photo = identity_for(path) if path else None
        self._preset_visibility_stamp = None
        self.preset_thumbnails.set_view(self._preset_photo, ())

    def _preset_spec(self, card):
        brand = self.brand.get()
        if getattr(self,'editor_enabled',False):
            from photo_edit import PhotoHistory
            data=self.user_presets[card[5:]] if card.startswith('user:') else None
            look=data['look'] if data else card
            style=self._preset_committed_styles.get(PhotoHistory.style_key(brand,look))
            controls=style['controls'] if style else data.get('controls',{}) if data else {}
            amount=style['amount'] if style else data['amount']/100 if data else 1.
            return spec_for(brand,look,controls=controls,amount=amount,base=self._preset_committed_base)
        if card.startswith('user:'):
            data = self.user_presets[card[5:]]
            return spec_for(brand, data['look'], data['exposure'], data.get('controls'),
                            data['amount'] / 100, data.get('post_balance', (0, 0)), data.get('post_settings'))
        return spec_for(brand, card)

    def bind_preset_cards(self):
        photo = self._preset_photo; store = self.preset_thumbnails
        self._preset_requests = {}; self.thumbnails = {}
        self._preset_visibility_stamp = None
        context = (photo, self.brand.get())
        for card in self.cards:
            if photo is None:
                label = self.cards[card][1]
                from filter_catalog import renderer_available
                spec = self._preset_spec(card)
                label.configure(image='', text='等待照片' if renderer_available(spec.brand, spec.look) else '尚未开放', width=15, height=5)
                label.image = None; label._style_source = None
                continue
            spec = self._preset_spec(card)
            from filter_catalog import renderer_available
            if not renderer_available(spec.brand, spec.look):continue
            key = store.choose(photo, card, spec, replace=getattr(self,'editor_enabled',False) or card.startswith('user:'))
            self._preset_requests[card] = key
            image = store.image(photo, card, key)
            if image is not None:
                store.remember(photo, card, key); self.thumbnails[card] = image
                self.set_card_image(card, image)
            elif context != self._preset_context:
                label = self.cards[card][1]
                label.configure(image='', text='等待生成', width=15, height=5)
                label.image = None; label._style_source = None
        self._preset_context = context
        self._request_visible_presets()

    def _request_visible_presets(self):
        # Only cards in/near the current viewport plus the selected card.
        keys = []; selected = self.active_preset or self.look.get()
        if selected in self._preset_requests:keys.append(self._preset_requests[selected])
        top = self.preset_canvas.canvasy(0); height = self.preset_canvas.winfo_height()
        stamp = (self._preset_photo, self.brand.get(), selected, top, height,
                 self.card_frame.winfo_height(), self._preset_card_layout)
        if stamp == self._preset_visibility_stamp:return
        self._preset_visibility_stamp = stamp
        for index, (card, (frame, label)) in enumerate(self.cards.items()):
            y = frame.winfo_y(); h = frame.winfo_height()
            visible = index < 8 if height <= 1 or h <= 1 else y + h >= top - 140 and y <= top + height + 140
            if visible and card in self._preset_requests:keys.append(self._preset_requests[card])
        self.preset_thumbnails.set_view(self._preset_photo, keys)

    def poll_preset_thumbnails(self):
        photo = self._preset_photo
        if photo is None:return
        # Working slider values do not change style cards. A committed base
        # will explicitly replace the view's cache revision at the UI boundary.
        self._request_visible_presets()
        # Reuse original decoded input if already available. Never use clean
        # AI pixels or a filter output for the style-card source.
        prepared = self.photo_denoise_store.get_source(photo)
        # Let a selected-photo render start/finish before starting another
        # card. AI progress alone is deliberately not a pause condition.
        paused = self.busy or bool(self.preview_started) or time.monotonic()<getattr(self,'_editor_interacting_until',0.)
        for key, image, error in self.preset_thumbnails.poll(paused, prepared):
            for card, wanted in self._preset_requests.items():
                if wanted != key:continue
                if image is not None:
                    self.preset_thumbnails.remember(photo, card, key)
                    self.thumbnails[card] = image; self.set_card_image(card, image)
                elif self.preset_thumbnails.image(photo, card, key) is None and card in self.cards:
                    self.cards[card][1].configure(text='预览失败 · 点击重试')
