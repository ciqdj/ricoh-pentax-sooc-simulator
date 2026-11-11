"""Photo-owned denoising state, independent of colour-render generations."""
from contextlib import contextmanager
from pathlib import Path
import time

from photo_denoise import PhotoDenoiseStore, key_for, working_input
from postprocess import validate as validate_post
import gpu_compute


ACTIVE = {'queued', 'preparing', 'running'}


class SourceDenoiseMixin:
    def init_source_denoise(self):
        self.photo_denoise_store = PhotoDenoiseStore()
        self.photo_denoise_values = {}
        self.photo_native_noise = {}
        self.source_denoise_keys = set()
        self._source_input_token = None
        self._source_status_stamp = None
        self._source_completed_keys = set()

    def photo_settings_key(self):
        path = getattr(self, 'current_path', None)
        return str(Path(path).resolve()) if path else None

    def current_photo_denoise(self):
        if getattr(self,'editor_enabled',False):
            values=self.editor_state['denoise']
            return {'denoise':values['amount'] if values['kind']=='ai' else 0.,
                    'denoise_model':values['model'],'noise_level':15.}
        return dict(self.photo_denoise_values.get(self.photo_settings_key(),
                    {'denoise': 0., 'denoise_model': 'dcu', 'noise_level': 15.}))

    def native_noise_values(self):
        if getattr(self,'editor_enabled',False):
            values=self.editor_state['denoise']
            return dict(values['native']) if values['kind']=='native' else {}
        return dict(self.photo_native_noise.get(self.photo_settings_key(), {}))

    def effective_overrides(self, look=None):
        from native_controls import NR_CONTROLS
        from filter_catalog import PENTAX
        values = dict(self.overrides.get(look or self.look.get(), {}))
        for name in NR_CONTROLS:values.pop(name, None)
        if self.brand.get() == PENTAX:
            from native_backend import LOOKS
            if (look or self.look.get()) in LOOKS:values.update(self.native_noise_values())
        return values

    def current_post(self):
        settings = validate_post(self.editor_state['post'] if getattr(self,'editor_enabled',False)
                                 else self.post_values.get((self.brand.get(), self.look.get())))
        settings.update(self.current_photo_denoise())
        return settings

    def render_post_settings(self):
        if getattr(self,'editor_enabled',False):return validate_post()
        return dict(self.current_post(), denoise=0.)

    def denoise_key(self):
        path = getattr(self, 'current_path', None)
        if not path:
            return None
        settings = self.current_photo_denoise()
        try:
            return key_for(path, settings['denoise_model'], 'auto')
        except OSError:
            return None

    def current_denoise(self):
        key = self.denoise_key()
        item = self.photo_denoise_store.get(key) if key is not None else None
        previous = self.denoise_result
        if item is None:
            # The selected photo owns a lease even if a background photo
            # evicts the shared LRU. key_for has rechecked the source identity.
            if key is not None and previous is not None and previous.get('key') == key:
                return previous
            self.denoise_result = None
            return None
        if previous is not None and previous.get('record') is item:
            return previous
        self.denoise_result = {'key': key, 'record': item, 'source': item.source,
                              'clean': item.rgb, 'alpha': item.alpha, 'seconds': item.seconds}
        return self.denoise_result

    def current_denoise_status(self):
        key = self.denoise_key()
        state = self.photo_denoise_store.status(key) if key is not None else {'state':'missing'}
        result = self.current_denoise()
        if result is not None:
            return dict(state, state='completed', stage='已完成', percent=100,
                        elapsed=result['seconds'], error='', error_code='')
        return state

    def denoise_idle(self):
        return not any(self.photo_denoise_store.status(key)['state'] in ACTIVE
                       for key in tuple(self.source_denoise_keys))

    def invalidate_denoise(self):
        # A colour/brand/render generation is not an AI input identity.
        if self.ready:
            self.refresh_denoise_controls()

    def source_snapshot(self):
        result = self.current_denoise()
        amount = self.current_photo_denoise()['denoise']
        return (self.current_path, result['record'] if result and amount else None, amount)

    def source_token(self, snapshot=None):
        path, result, amount = snapshot or self.source_snapshot()
        return (str(path), (result.key,getattr(result,'run_id',id(result))) if result is not None else None,
                amount if result is not None else 0.)

    @staticmethod
    @contextmanager
    def source_input(snapshot, cancelled=None):
        path, result, amount = snapshot
        if result is None:
            yield path
        else:
            # The snapshot retains the Result lease until all filter readers
            # have stopped, even when the store has evicted its cache entry.
            with working_input(result, amount, cancelled) as working:
                yield working

    def refresh_source_render(self):
        if not self.ready or self.closed or not self.current_path or self.busy:
            return
        if getattr(self,'editor_enabled',False):
            snapshot=self.source_snapshot();token=self.source_token(snapshot)
            if token!=self._source_input_token:
                self.request_render()
            return
        from filter_catalog import renderer_available
        if not renderer_available(self.brand.get(),self.look.get()):return
        snapshot = self.source_snapshot()
        token = self.source_token(snapshot)
        if token == self._source_input_token:
            return
        self._source_input_token = token
        self.generation += 1
        self.request_revision += 1
        self.native_cache = {}; self.full_cache = None; self.detail_cache = None
        # Style cards belong to the original photo, not this clean-input
        # generation. Keep their images and controls while the main view reloads.
        self.preview_times = {}
        self.cache_evictions = __import__('queue').SimpleQueue()
        self.worker_failed = False; self.preview_error = ''
        self.reset_histogram()
        if self.pending:
            self.root.after_cancel(self.pending); self.pending = None
        self.requested = self.signature()
        self.preview_started = time.monotonic(); self.preview_stage = '正在更新照片'
        self.photo_started = time.monotonic()
        self.draw()
        self.pool.submit(self.load_worker, self.current_path, self.generation,
                         self.cache_evictions, self.brand.get(), snapshot)

    def denoise_changed(self, value=None):
        if getattr(self,'editor_enabled',False):return self.editor_denoise_changed(value)
        if not self.ready or self.busy:
            return
        from workbench_app import DENOISE_MODELS
        previous = self.current_photo_denoise()
        settings = dict(previous, denoise=round(self.denoise_amount.get()),
                        denoise_model=next(k for k,v in DENOISE_MODELS.items()
                                           if v == self.denoise_model.get()))
        self.photo_denoise_values[self.photo_settings_key()] = settings
        self.denoise_error = ''
        if self.denoise_detail is not None:
            self.denoise_detail.set_source_mode('input')
        if settings['denoise'] and (settings['denoise_model'] != previous['denoise_model']
                                    or not previous['denoise']):
            self.start_denoise()
        self.refresh_source_render()
        self.refresh_denoise_controls()

    def toggle_denoise(self):
        self.denoise_amount.set(100 if self.denoise_enabled.get() else 0)
        self.denoise_changed()

    def start_denoise(self):
        if getattr(self,'editor_enabled',False) and (self.editor_state['denoise']['kind']!='ai' or self.editor_state['denoise']['amount']<=0):return
        if self.busy:
            self.status.set('导出完成后可开始新的降噪任务。'); return
        if not self.current_path:
            self.status.set('请先选择照片。'); return
        if getattr(self,'editor_enabled',False):self.editor_history_only=False
        settings = self.current_photo_denoise()
        if not settings['denoise'] and not getattr(self,'editor_enabled',False):
            settings['denoise'] = 100.
            self.photo_denoise_values[self.photo_settings_key()] = settings
            self.denoise_amount.set(100)
        try:
            key = self.photo_denoise_store.request(self.current_path, settings['denoise_model'],
                                                   'auto')
            self.source_denoise_keys.add(key)
            self.denoise_error = ''
            self._source_status_stamp = None
            self.show_denoise_stage('正在准备输入照片…')
            if self.denoise_detail is not None:
                self.denoise_detail.set_source_mode('input')
            self.refresh_denoise_controls()
        except Exception as exc:
            self.denoise_error = str(exc)
            self.denoise_state.set('降噪未能开始：' + str(exc))

    def _clear_denoise_result(self, key):
        self.photo_denoise_store.discard(key)
        if self.denoise_result is not None and self.denoise_result['key']==key:
            self.denoise_result=None
        self.source_denoise_keys.discard(key)
        self._source_completed_keys.discard(key)
        self._source_status_stamp=None
        if self.denoise_detail is not None:self.denoise_detail.clear_input_cache()
        self.refresh_source_render()

    def regenerate_denoise(self):
        if not self.current_path or self.busy:return
        settings=self.current_photo_denoise()
        if not settings['denoise'] or (getattr(self,'editor_enabled',False) and self.editor_state['denoise']['kind']!='ai'):
            return
        key=self.denoise_key()
        if key is None:return
        self._clear_denoise_result(key)
        self.denoise_state.set('已清除旧降噪结果，正在重新生成…')
        self.start_denoise()

    def discard_denoise(self):
        if not self.current_path or self.busy:return
        key=self.denoise_key()
        if key is None:return
        if getattr(self,'editor_enabled',False) and self.editor_state['denoise']['kind']=='ai':
            from copy import deepcopy
            state=deepcopy(self.editor_state)
            state['denoise']['kind']='off'
            if not self.editor_record(state,'丢弃降噪结果'):return
        elif not getattr(self,'editor_enabled',False):
            settings=dict(self.current_photo_denoise(),denoise=0.)
            self.photo_denoise_values[self.photo_settings_key()]=settings
            self.denoise_amount.set(0)
        self._clear_denoise_result(key)
        self.denoise_error=''
        self.denoise_state.set('已丢弃降噪结果')
        self.refresh_denoise_controls()

    def cancel_denoise(self):
        current = self.denoise_key()
        keys = [current] if current and self.photo_denoise_store.status(current)['state'] in ACTIVE else [
            key for key in self.source_denoise_keys if self.photo_denoise_store.status(key)['state'] in ACTIVE]
        for key in keys:
            self.photo_denoise_store.cancel(key)
        self.denoise_error = '已取消'
        self.stop_denoise_preparation()
        self.denoise_state.set('正在停止…' if keys else '已取消')

    def refresh_denoise_controls(self):
        if not self.ready:
            return
        settings = self.current_photo_denoise(); amount = settings['denoise']
        key = self.denoise_key()
        state = self.current_denoise_status()
        active = state['state'] in ACTIVE
        result = self.current_denoise()
        ai_selected=not getattr(self,'editor_enabled',False) or self.editor_state['denoise']['kind']=='ai'
        self.denoise_enabled.set(bool(amount))
        self.denoise_label.set('DCU 原厂' if getattr(self,'editor_enabled',False) and self.editor_state['denoise']['kind']=='native' else '关闭' if not amount else
                              f'结果混合 {amount:.0f}%' + ('（已应用）' if result else ''))
        can_act=bool(self.current_path and not self.busy and ai_selected)
        self.denoise_start_button.configure(state='normal' if can_act and amount and not active and result is None else 'disabled')
        self.denoise_regenerate_button.configure(state='normal' if can_act and amount and not active and result is not None else 'disabled')
        self.denoise_discard_button.configure(state='normal' if can_act and (result is not None or state['state']!='missing') else 'disabled')
        self.denoise_cancel_button.configure(state='normal' if not self.denoise_idle() else 'disabled')
        self.denoise_combo.configure(state='disabled' if self.busy else 'readonly')

    def poll_denoise(self):
        if not self.ready:
            return
        # Only queued/running jobs need global polling. Completed records are
        # found by the selected photo key; do not stat an ever-growing album
        # sixty times a second after those jobs have finished.
        self.source_denoise_keys.intersection_update(
            key for key in tuple(self.source_denoise_keys)
            if self.photo_denoise_store.status(key)['state'] in ACTIVE)
        key = self.denoise_key()
        state = self.current_denoise_status()
        active = state['state'] in ACTIVE
        if not active:
            other = next((k for k in self.source_denoise_keys
                          if self.photo_denoise_store.status(k)['state'] in ACTIVE), None)
            if other is not None:
                state = self.photo_denoise_store.status(other)
                state = dict(state, stage=Path(other.identity.path).name + ' · ' + state.get('stage','后台处理中'))
                active = True
        stamp = (state['state'], state.get('stage'), state.get('percent'), int(state.get('elapsed',0)),
                 self.current_photo_denoise()['denoise'], key)
        if stamp != self._source_status_stamp:
            self._source_status_stamp = stamp
            if active:
                text = state.get('stage') or '正在处理…'
                percent = state.get('percent')
                if percent is not None and '%' not in text:
                    text += f' {min(99,percent):.0f}%'
                self.show_denoise_stage(text)
                if percent is None:
                    self.denoise_state.set(text + f'    {int(state.get("elapsed",0))} 秒')
            else:
                self.stop_denoise_preparation()
                if state['state'] == 'completed' and self.current_denoise() is not None:
                    self.denoise_progress['value'] = 100; self.denoise_progress_text.set('100%')
                    self.denoise_state.set(f'已完成    {self.current_denoise()["seconds"]:.1f} 秒')
                elif state['state'] in ('failed', 'cancelled'):
                    self.denoise_progress['value'] = 0
                    self.denoise_progress_text.set('失败' if state['state']=='failed' else '—')
                    self.denoise_state.set(state.get('error') or '已取消')
                else:
                    self.denoise_progress['value'] = 0; self.denoise_progress_text.set('—')
                    self.denoise_state.set('')
            self.refresh_denoise_controls()
        # Only the selected photo's input changes its filter render. A result
        # finishing for another photo remains cached without touching the UI.
        self.refresh_source_render()

    def needs_full_detail(self):
        return self.want_detail or bool(self.denoise_detail and self.denoise_detail.needs_detail)
