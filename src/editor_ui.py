"""Photo-owned working adjustments and inline history previews.

History stores parameters only. Selecting a historical step previews it;
recording an edit after that step is the only place that asks to branch.
Legacy commit groups are read as flat steps; there is no commit/undo UI.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json, time, queue, threading, re
from ui_language import tk, ttk, messagebox
from native_controls import NR_CONTROLS
from native_noise_panel import NativeNoisePanel
from postprocess import validate as validate_post
from ui_controls import Tooltip


DENOISE_CHOICES = {'off': '关闭', 'native': 'DCU 原厂', 'nafnet': 'NAFNet',
                   'restormer': 'Restormer'}


def _clock(value):
    try:
        if isinstance(value, (int, float)):return datetime.fromtimestamp(value).strftime('%H:%M:%S')
        return datetime.fromisoformat(str(value)).strftime('%H:%M:%S')
    except (ValueError, TypeError, OSError):return str(value or '')[-8:]


class EditorNativeNoisePanel(NativeNoisePanel):
    """The existing four controls under the single exclusive method selector."""
    def __init__(self, app, parent):
        super().__init__(app, parent)
        self.toggle.pack_forget()
        for widget in self.body.winfo_children():
            if isinstance(widget, ttk.Button):widget.configure(text='恢复默认')

    def available(self):
        a = self.app
        from native_backend import LOOKS
        return bool(a.ready and a.current_path and not a.busy and
                    a.editor_state['denoise']['kind'] == 'native' and
                    a.editor_state['filter']['brand']=='pentax' and a.editor_state['filter']['look'] in LOOKS)

    def reset(self):
        if not self.available():return
        state = deepcopy(self.app.editor_state); state['denoise']['native'] = {}
        self.app.editor_record(state, '恢复 DCU 降噪默认')

    def changed(self, field=None, value=None):
        if self.syncing or not self.available():return
        if field is not None:self.values[field].set(round(float(value)))
        controls = {key: max(meta['min'], min(meta['max'], round(self.values[key].get())))
                    for key, meta in NR_CONTROLS.items()}
        state = deepcopy(self.app.editor_state); state['denoise']['native'] = controls
        label=(NR_CONTROLS[field]['name_zh']+' '+str(controls[field])) if field is not None else 'DCU 降噪调整'
        self.app.editor_record(state,label,coalesce_key='native-noise-'+str(field))

    def sync(self):
        a = self.app
        if not a.ready or not getattr(a, 'editor_enabled', False):return
        active = a.editor_state['denoise']['kind'] == 'native'
        controls = a.editor_state['denoise']['native'] if active else {}
        report = getattr(a, 'editor_report', {})
        factory = report.get('factory_noise_reduction', {})
        numbers = {key: controls.get(key, factory.get(key, {}).get('value', meta['default']))
                   for key, meta in NR_CONTROLS.items()}
        signature = (active, self.available(), tuple(numbers.items()))
        if signature == self.key:return
        self.key = signature; self.syncing = True
        try:
            self.enabled.set(active)
            for key, value in numbers.items():
                self.values[key].set(value); self.labels[key].set(str(value))
                self.sliders[key].configure(state='normal' if self.available() else 'disabled')
            if active:self.body.pack(fill='x')
            else:self.body.pack_forget()
            self.note.set('原厂降噪使用 100% 滤镜浓淡' if active else '')
        finally:self.syncing = False


class EditorMixin:
    def init_editor_state(self):
        from app_paths import CONFIG
        from photo_edit import PhotoEditStore, default_state
        self.photo_edit_store = PhotoEditStore(CONFIG / 'photo_edits')
        self.editor_state = default_state(); self.editor_history = None
        self.editor_enabled = False; self.editor_loading = False; self.editor_pending = False
        self.editor_history_only = False
        self.history_popup = None; self.history_tree = None; self.history_items = {}
        self.task_popup = None; self.task_tree = None; self._task_stamp = None
        self.editor_report = {}; self._editor_commit_serial = None
        self._editor_requested = None; self._editor_worker_token = None

    def build_editor_toolbar(self, right):
        toolbar = ttk.Frame(right)
        toolbar.pack(fill='x', before=self.tone_panel, pady=(0, 8))
        style = ttk.Style(self.root); style.configure('Editor.TButton', padding=(5, 4))
        style.configure('Editor.Treeview',background='#2c2c2e',fieldbackground='#2c2c2e',foreground='#f2f2f7',rowheight=24)
        style.map('Editor.Treeview',background=[('selected','#0a84ff'),('!selected','#2c2c2e')],
                  foreground=[('selected','#ffffff'),('!selected','#f2f2f7')],fieldbackground=[('!disabled','#2c2c2e')])
        style.configure('Editor.Treeview.Heading',background='#3a3a3c',foreground='#f2f2f7')
        self.restore_button = ttk.Button(toolbar, text='恢复原始', style='Editor.TButton', command=self.editor_restore_original)
        self.history_button = ttk.Button(toolbar, text='历史 ▾', style='Editor.TButton', command=self.open_history_menu)
        for i, button in enumerate((self.restore_button, self.history_button)):
            button.grid(row=0, column=i, sticky='ew', padx=(0, 4) if i == 0 else 0)
            toolbar.columnconfigure(i, weight=1)
        self.history_panel=ttk.Frame(toolbar)
        self.history_panel.grid(row=1,column=0,columnspan=2,sticky='ew',pady=(5,0))
        self.history_panel.grid_remove();self.history_expanded=False
        self.editor_toolbar = toolbar
        header = self.root.winfo_children()[0]
        self.task_box=ttk.Frame(header);self.task_box.pack(side='right',padx=8)
        summaries=ttk.Frame(self.task_box);summaries.pack(side='left')
        self.task_summaries=[]
        for _ in range(2):
            value=tk.StringVar(master=self.root,value='')
            label=ttk.Label(summaries,textvariable=value,wraplength=345,justify='left',font=('Microsoft YaHei UI',8),style='Hint.TLabel')
            label.pack(anchor='w');Tooltip(label,value.get)
            self.task_summaries.append(value)
        self.task_text = tk.StringVar(master=self.root, value='后台任务')
        self.task_button = ttk.Button(self.task_box, textvariable=self.task_text, style='Editor.TButton', command=self.open_task_menu)
        self.task_button.pack(side='right',padx=(6,0))
        self.editor_enabled = True
        self.editor_sync_toolbar()

    def bind_editor_shortcuts(self, widget):
        # Photo history has no undo/redo shortcut. Entry/Text class bindings
        # remain untouched, including their own text-editing commands.
        return

    def editor_shortcut(self, event, redo=False):
        return

    def editor_sync_toolbar(self):
        if not getattr(self, 'editor_enabled', False):return
        enabled = self.editor_history is not None and not self.busy
        for widget in (self.restore_button, self.history_button):
            widget.configure(state='normal' if enabled else 'disabled')

    def editor_open_photo(self, path):
        pending = deepcopy(self.editor_state) if self.editor_pending else None
        self.editor_history = self.photo_edit_store.for_photo(path)
        self.editor_pending = False
        self.editor_state = self.editor_history.current
        if pending is not None:self.editor_record(pending, '应用初始设置', render=False)
        self.apply_editor_state(self.editor_history.current, render=False)
        self.editor_sync_toolbar(); self.refresh_history_tree(); self.editor_sync_committed_cards(force=True)

    def editor_record(self, state, label, coalesce_key=None, render=True):
        if self.editor_loading or self.busy:return False
        from photo_edit import BranchRequired, normalize,describe_change,fingerprint
        state = normalize(state)
        if state==self.editor_state:return False
        # One bounded preview stream while controls move. Full-size native
        # processing starts only after the latest edit has settled.
        self._editor_interacting_until=time.monotonic()+.25
        if coalesce_key in ('basic','mixer','filter-controls'):
            def changed(a,b,prefix=''):
                if isinstance(a,dict) and isinstance(b,dict):
                    return [path for name in b for path in changed(a.get(name),b[name],prefix+'/'+name)]
                if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b):
                    return [path for i,(old,new) in enumerate(zip(a,b)) for path in changed(old,new,prefix+'/'+str(i))]
                return [] if a==b else [prefix]
            coalesce_key+=':'+fingerprint(changed(self.editor_state,state))
        if label in ('基础调整','颜色混合器','色彩平衡','明暗曲线','滤镜 / 曝光调整','DCU 降噪调整','降噪调整'):
            label=describe_change(self.editor_state,state)
        if self.editor_history is None:
            self.editor_state = state; self.editor_pending = True
        else:
            try:self.editor_history.record(state, label, coalesce_key=coalesce_key)
            except BranchRequired:
                if not messagebox.askyesno('继续编辑？', '继续编辑会删除当前历史位置之后的步骤。', parent=self.root):
                    self.apply_editor_state(self.editor_history.current, render=False, history_preview=self.editor_history_only); return False
                self.editor_history.record(state, label, allow_branch=True, coalesce_key=coalesce_key)
            self.editor_state = self.editor_history.current
        self.apply_editor_state(self.editor_state, render=render)
        self.refresh_history_tree(); self.editor_sync_toolbar(); return True

    def apply_editor_state(self, state, render=True, history_preview=False):
        previous_look = (self.brand.get(),self.look.get())
        self.editor_state = deepcopy(state); self.editor_loading = True
        self.editor_history_only=bool(history_preview)
        try:
            current = state['filter']; brand = current['brand']; look = current['look'] or ''
            previous_brand = self.brand.get()
            self.brand.set(brand); self.look.set(look); self.amount.set(current['amount'] * 100)
            self.exposure.set(state['exposure']); self.active_preset = None
            self.overrides = self.brand_overrides.setdefault(brand, {})
            if look:self.overrides[look] = dict(current['controls'])
            if brand == 'custom' and look:
                try:self.custom_brand_id = self.custom_library.lookup(look)['brand_id']
                except ValueError:pass
            post = validate_post(state['post'])
            if self.balance_pad:self.balance_pad.set(state['balance'])
            if self.tone_panel:self.tone_panel.set(post['tone'])
            if self.basic_panel:self.basic_panel.set(post['basic'], post['tone']['regions'])
            if self.mixer_panel:self.mixer_panel.set(post['mixer'])
            choice = state['denoise']; selected = choice['kind'] if choice['kind'] != 'ai' else choice['model']
            self.denoise_model.set(DENOISE_CHOICES[selected]); self.denoise_amount.set(choice['amount'])
            self.denoise_enabled.set(choice['kind'] == 'ai')
            self.photo_denoise_values[self.photo_settings_key()] = {
                'denoise': choice['amount'] if choice['kind'] == 'ai' else 0.,
                'denoise_model': choice['model'], 'noise_level': 15.}
            self.photo_native_noise[self.photo_settings_key()] = dict(choice['native']) if choice['kind'] == 'native' else {}
            self.amount_text.set(f'浓淡   {current["amount"] * 100:.0f}%')
            self.ev_text.set(f'曝光补偿   {state["exposure"]:+.2f} EV')
            from filter_catalog import catalog,GR_ENGLISH
            name=catalog(brand).get(look,{}).get('name',look)
            self.preset_text.set('原图' if not look else name if brand=='custom' else GR_ENGLISH.get(look,look)+'\n'+name)
            if previous_brand != brand:
                self.group.set('全部'); self.update_custom_controls(); self.build_cards()
            if self.ready:
                if previous_look!=(brand,look):self.build_controls()
                else:
                    for name,(variable,widget) in self.control_widgets.items():
                        value=current['controls'].get(name,self.factory.get(look,{}).get('factory_defaults',{}).get(name))
                        if value is not None and variable.get()!=value:variable.set(value)
                self.highlight(); self.update_brand_controls()
        finally:self.editor_loading = False
        if render and self.current_path and self.pending is None:self.pending=self.root.after(40,self.request_render)
        if not self.editor_history_only and state['denoise']['kind'] == 'ai' and state['denoise']['amount']>0 and self.current_path and self.current_denoise() is None:
            self.start_denoise()
        if self.editor_history_missing_denoise():self.status.set('此步骤的降噪结果未缓存，需重新计算')

    def editor_history_missing_denoise(self):
        choice=self.editor_state['denoise']
        return bool(self.editor_history_only and choice['kind']=='ai' and choice['amount']>0 and self.current_denoise() is None)

    def editor_manual_changed(self, label='基础调整', coalesce_key='manual', **changes):
        if not self.ready or self.editor_loading or self.busy:return
        state = deepcopy(self.editor_state); state.update(changes)
        self.editor_record(state, label, coalesce_key=coalesce_key)

    def editor_filter_selected(self, key):
        from filter_catalog import CUSTOM, catalog
        state = deepcopy(self.editor_state)
        if key.startswith('user:'):
            saved = self.user_presets[key[5:]]
            # Choosing any preset changes only the look. Older saved presets
            # contain photo-wide edits; importing those here silently erased
            # the current photo's exposure, curve and colour adjustments.
            state['filter'] = {'brand': saved.get('brand', 'pentax'), 'look': saved['look'],
                               'amount': saved['amount'] / 100,
                               'controls': {k: v for k, v in saved.get('controls', {}).items() if k not in NR_CONTROLS}}
            label = '预设 · ' + key[5:]
        else:
            from photo_edit import PhotoHistory
            committed=self.editor_history.committed_styles if self.editor_history is not None else {}
            style=committed.get(PhotoHistory.style_key(self.brand.get(),key)) if key else None
            state['filter'] = deepcopy(style) if style is not None else {'brand': self.brand.get(), 'look': key or None, 'amount': 1., 'controls': {}}
            label = '移除滤镜' if not key else '滤镜 · ' + catalog(self.brand.get())[key]['name']
        from native_backend import LOOKS
        if state['denoise']['kind']=='native' and (state['filter']['brand']!='pentax' or state['filter']['look'] not in LOOKS):
            state['denoise']['kind']='off';label+=' · 关闭原厂降噪'
        elif state['denoise']['kind']=='native':state['filter']['amount']=1.
        return self.editor_record(state, label)

    def editor_filter_changed(self):
        if self.editor_loading:return
        state = deepcopy(self.editor_state)
        state['exposure'] = float(self.exposure.get())
        state['filter']['amount'] = float(self.amount.get()) / 100
        if state['denoise']['kind']=='native':state['filter']['amount']=1.
        state['filter']['controls'] = {k: v for k, v in self.overrides.get(self.look.get(), {}).items() if k not in NR_CONTROLS}
        self.editor_record(state, '滤镜 / 曝光调整', coalesce_key='filter-controls')

    def editor_denoise_changed(self, value=None):
        if not self.ready or self.editor_loading or self.busy:return
        selected = next(key for key, label in DENOISE_CHOICES.items() if label == self.denoise_model.get())
        previous_key=self.denoise_key()
        state = deepcopy(self.editor_state); choice = state['denoise']
        from native_backend import LOOKS
        if selected=='native' and (state['filter']['brand']!='pentax' or state['filter']['look'] not in LOOKS):
            old=choice['kind'] if choice['kind']!='ai' else choice['model']
            self.denoise_model.set(DENOISE_CHOICES[old]);self.status.set('选择宾得滤镜后可使用 DCU 原厂降噪。');return
        choice['kind'] = selected if selected in ('off', 'native') else 'ai'
        if selected=='native':state['filter']['amount']=1.
        if choice['kind'] == 'ai':choice['model'] = selected
        choice['amount'] = float(self.denoise_amount.get())
        previous=self.editor_state['denoise']
        label='降噪调整' if (previous['kind'],previous['model'])==(choice['kind'],choice['model']) else '降噪 · '+DENOISE_CHOICES[selected]
        changed=self.editor_record(state,label,coalesce_key='denoise')
        if changed and choice['kind']!='ai' and previous_key is not None:self.photo_denoise_store.cancel(previous_key)
        if self.denoise_detail is not None:self.denoise_detail.set_source_mode('native' if selected == 'native' else 'input')

    def editor_undo(self):
        if self.editor_history is None or self.busy:return
        self.editor_history.undo(); self.apply_editor_state(self.editor_history.current, history_preview=True)
        self.refresh_history_tree(); self.editor_sync_toolbar()

    def editor_redo(self):
        if self.editor_history is None or self.busy:return
        self.editor_history.redo(); self.apply_editor_state(self.editor_history.current, history_preview=True)
        self.refresh_history_tree(); self.editor_sync_toolbar()

    def editor_restore_original(self):
        if self.editor_history is None or self.busy:return
        from photo_edit import BranchRequired
        try:self.editor_history.restore_original()
        except BranchRequired:
            if not messagebox.askyesno('恢复原始？','恢复原始会删除当前历史位置之后的步骤。',parent=self.root):return
            self.editor_history.restore_original(allow_branch=True)
        self.apply_editor_state(self.editor_history.current)
        self.refresh_history_tree();self.editor_sync_toolbar()

    def editor_commit(self):
        if self.editor_history is None or self.busy:return
        from photo_edit import BranchRequired
        try:self.editor_history.commit()
        except BranchRequired:
            if not messagebox.askyesno('提交当前步骤？', '提交会删除当前历史位置之后的步骤。', parent=self.root):return
            self.editor_history.commit(allow_branch=True)
        self.editor_sync_committed_cards(); self.refresh_history_tree(); self.status.set('已提交')

    def editor_sync_committed_cards(self, force=False):
        if self.editor_history is None:return
        serial = self.editor_history.commit_serial
        if not force and serial == self._editor_commit_serial:return
        self._editor_commit_serial = serial
        update = getattr(self, 'set_committed_preset_base', None)
        if update is not None:update(self.editor_history.committed_base,
                                     self.editor_history.committed_styles, serial)

    def editor_switch_brand(self, brand, force=False):
        from filter_catalog import BRANDS
        if brand not in BRANDS:raise ValueError('Unknown filter brand')
        if self.busy or self.custom_importing or (brand==self.brand.get() and not force):return
        state=deepcopy(self.editor_state)
        state['filter']={'brand':brand,'look':None,'amount':1.,'controls':{}}
        label='切换至 '+BRANDS[brand]
        if state['denoise']['kind']=='native':state['denoise']['kind']='off';label+=' · 关闭原厂降噪'
        self.editor_record(state,label)
        self.group.set('全部');self.update_custom_controls();self.build_cards();self.preset_canvas.yview_moveto(0)

    def editor_signature(self):
        from photo_edit import fingerprint
        state=self.editor_state;selected=state['filter']
        record=self.current_denoise() if state['denoise']['kind']=='ai' else None
        stamp=fingerprint({'state':state,'input':str(record['key']) if record else None})
        return (selected['look'] or '',state['exposure'],tuple(sorted(selected['controls'].items())),selected['amount'],stamp)

    def editor_request_render(self):
        self.pending=None
        if not self.current_path or self.closed or self.busy:return
        signature=self.editor_signature();self.requested=signature;self.request_revision+=1;self.preview_error=''
        held=self.current_denoise() if self.editor_state['denoise']['kind']=='ai' else None
        self._editor_requested=(signature,deepcopy(self.editor_state),held['record'] if held else None)
        if signature in self.native_cache:self.show_result(*self.native_cache[signature])
        else:
            self.preview_started=time.monotonic();self.preview_stage='正在更新，保留上一帧';self.draw()
        if self._editor_worker_token!=self.generation or self.worker_failed:
            self._editor_worker_token=self.generation;self.worker_failed=False
            self.pool.submit(self.load_editor_worker,self.current_path,self.generation,self.cache_evictions)

    def editor_file_selected(self,path):
        from photo_denoise import identity_for
        if path==self.current_path and self._preset_photo==identity_for(path):return
        if self.pending:self.root.after_cancel(self.pending);self.pending=None
        self.generation+=1;self.current_path=self.path=path;self.native_cache={};self.factory={}
        self.detail_cache=None;self.full_cache=None;self.viewport.reset();self._editor_requested=None
        self.cache_evictions=queue.SimpleQueue();self.worker_failed=False
        self.before=self.after=None;self.preview_error='';self.preview_data=None
        self.photo_started=time.monotonic();self.preview_times={};self.preview_started=0
        self.preset_photo_selected(path);self.reset_histogram()
        self.editor_open_photo(path)
        self.build_cards();self.build_controls();self.update_brand_controls();self.export_changed()
        self.request_render();self.draw()

    def load_editor_worker(self,path,token,evictions=None):
        from editor_pipeline import EditorSession
        from native_app import preview
        from native_backend import RenderCancelled
        from tone_tools import histogram
        completed={};failed={};evictions=self.cache_evictions if evictions is None else evictions
        try:
            with EditorSession(path,self.photo_denoise_store,self.custom_library) as session:
                while not self.closed and token==self.generation:
                    try:
                        while True:
                            generation,old=evictions.get_nowait()
                            if generation==token:completed.pop(old,None);failed.pop(old,None)
                    except queue.Empty:pass
                    request=self._editor_requested
                    if self.busy or request is None:time.sleep(.025);continue
                    sig,state,record=request;revision=self.request_revision
                    needs_full=self.needs_full_detail()
                    interacting=time.monotonic()<getattr(self,'_editor_interacting_until',0.)
                    detail_ready=self.detail_cache is not None and self.detail_cache[:2]==(token,sig)
                    level=completed.get(sig,0)
                    if interacting:
                        if level>=1:time.sleep(.012);continue
                        target=(480,360);display_level=1
                    elif level<2:
                        target=(960,720);display_level=2
                    elif needs_full and not detail_ready:
                        target=None;display_level=3
                    else:time.sleep(.025);continue
                    full=target is None
                    if failed.get(sig)==revision:time.sleep(.025);continue
                    def cancelled():
                        current=self._editor_requested
                        obsolete=(self.closed or token!=self.generation or self.busy or current is None or current[0][0]!=sig[0])
                        # Let a small intermediate frame finish; restarting
                        # it for every pointer event would starve the display.
                        # A stale full-photo job must yield to current edits.
                        return obsolete or (full and (current[0]!=sig or
                            time.monotonic()<getattr(self,'_editor_interacting_until',0.)))
                    def progress(text):self.events.put(('native_stage',(token,sig,text)))
                    try:
                        before,rgb,alpha,report=session.render(state,maxsize=target,
                            result=record,cancelled=cancelled,progress=progress)
                        if cancelled():raise RenderCancelled()
                        if full and sig==self.requested:self.detail_cache=(token,sig,before,rgb,alpha)
                        before_image=preview(before,alpha);after_image=preview(rgb,alpha)
                        report=dict(report,_display_histogram=histogram(after_image),
                                    _editor_display_level=display_level,_editor_interim=display_level==1)
                        completed[sig]=display_level
                        if len(completed)>24:completed.pop(next(iter(completed)))
                        self.events.put(('native_result',(token,sig,(before_image,after_image,report))))
                        self.events.put(('native_stats',(token,{'decode_count':session.decode_count,'max_parallel':1})))
                    except RenderCancelled:pass
                    except Exception as exc:failed[sig]=revision;self.events.put(('native_error',(token,sig,str(exc))))
        except RenderCancelled:pass
        except Exception as exc:self.events.put(('native_error',(token,None,str(exc))))

    def queue_post(self,image):
        if not self.editor_enabled:return super().queue_post(image)
        # EditorSession already applied manual edits before the filter. The
        # fit display and full viewport must not apply those edits a second time.
        self.post_revision+=1;self.post_completed=self.post_revision
        self.post_request=None;self.post_running=None;self.after=image
        hist=self.factory.get(self.requested[0],{}).get('_display_histogram') if self.requested else None
        if hist is not None:self.output_histogram=(self.generation,image,hist)
        self.draw()

    def editor_start_export(self):
        if self.busy or self.custom_importing or not self.files:return
        from photo_formats import FORMAT_CHOICES
        import gpu_compute
        try:
            quality=int(self.quality.get());bits=int(self.bits.get())
            assert 80<=quality<=100 and bits in (8,16) and self.output.get().strip()
        except (ValueError,tk.TclError,AssertionError):
            messagebox.showerror('导出设置','检查输出文件夹、位深和 JPEG 质量。',parent=self.root);return
        states={str(Path(path).resolve()):self.photo_edit_store.for_photo(path).current for path in self.files}
        held=self.current_denoise();records={self.photo_settings_key():held['record']} if held else {}
        self.busy=True;self.cancel_export.clear();self.export_cancelled=False;self.progress['value']=0
        self.video_run.configure(state='disabled');self.update_brand_controls()
        args=(tuple(self.files),Path(self.output.get()),states,FORMAT_CHOICES[self.format.get()],quality,bits,gpu_compute.status()['mode'],records)
        self.export_thread=threading.Thread(target=self.editor_export_worker,args=args,daemon=True);self.export_thread.start()

    def editor_export_worker(self,files,folder,states,fmt,quality,bits,backend,records):
        from editor_pipeline import save
        from photo_formats import destination_for
        from native_backend import RenderCancelled
        outputs=[];errors=[]
        for i,path in enumerate(files):
            try:
                if self.cancel_export.is_set():raise RenderCancelled()
                state=states[str(Path(path).resolve())];look=state['filter']['look'] or 'Original'
                name=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',look)
                destination=destination_for(path,folder,name,fmt)
                def progress(text):self.events.put(('progress',(100*i/len(files),f'{i+1}/{len(files)} · {text}')))
                report=save(self.photo_denoise_store,path,destination,state,jpeg_quality=quality,bit_depth=bits,
                    backend_mode=backend,progress=progress,cancelled=self.cancel_export.is_set,
                    prepared_result=records.get(str(Path(path).resolve())),library=self.custom_library)
                outputs.append({'file':str(destination),'format':report['output_format'],'bits':report['output_bits']})
            except RenderCancelled:self.export_cancelled=True;break
            except Exception as exc:errors.append(Path(path).name+'：'+str(exc))
        self.events.put(('done',(len(outputs),errors,outputs)))

    def _editor_popup(self, attr, title, anchor, width=365, height=360):
        previous = getattr(self, attr, None)
        if previous is not None and previous.winfo_exists():previous.destroy(); setattr(self, attr, None); return None
        popup = tk.Toplevel(self.root); popup.withdraw(); popup.title(title); popup.transient(self.root)
        popup.overrideredirect(True); popup.configure(bg='#48484a')
        popup.geometry(f'{width}x{height}+{max(0,anchor.winfo_rootx()+anchor.winfo_width()-width)}+{anchor.winfo_rooty()+anchor.winfo_height()+3}')
        top = ttk.Frame(popup, padding=(6, 3)); top.pack(fill='x')
        ttk.Label(top, text=title).pack(side='left')
        ttk.Button(top, text='×', width=3, command=popup.destroy).pack(side='right')
        popup.bind('<Escape>', lambda event:popup.destroy())
        # No grab or focus transfer: a dropdown must not block other windows.
        # Root bindings are temporary and removed when this popup is destroyed.
        def dismiss(event=None):
            if popup.winfo_exists():popup.destroy()
            return 'break'
        def outside(event):
            widget=event.widget
            while widget is not None:
                if widget is popup or widget is anchor:return
                widget=getattr(widget,'master',None)
            if popup.winfo_exists():popup.destroy()
        bindings=[('<Escape>',self.root.bind('<Escape>',dismiss,add='+')),
                  ('<ButtonPress-1>',self.root.bind('<ButtonPress-1>',outside,add='+'))]
        def destroyed(event):
            if event.widget is not popup:return
            for sequence,identifier in bindings:
                if identifier:
                    try:self.root.unbind(sequence,identifier)
                    except tk.TclError:pass
            if getattr(self,attr,None) is popup:setattr(self,attr,None)
        popup.bind('<Destroy>',destroyed,add='+')
        setattr(self, attr, popup); return popup

    def _show_editor_popup(self,popup):
        popup.deiconify();popup.update_idletasks()
        if __import__('os').name=='nt':
            import ctypes
            from ctypes import wintypes as W
            user=ctypes.WinDLL('user32',use_last_error=True)
            user.GetAncestor.argtypes=[W.HWND,W.UINT];user.GetAncestor.restype=W.HWND
            user.SetWindowPos.argtypes=[W.HWND,W.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,W.UINT]
            user.SetWindowPos.restype=W.BOOL
            # Tk's borderless transient may map beneath its owner. Raise the
            # native wrapper without activation, focus, or global topmost state.
            if not user.SetWindowPos(user.GetAncestor(popup.winfo_id(),2),0,0,0,0,0,0x0013):
                raise ctypes.WinError(ctypes.get_last_error())
        else:popup.lift(self.root)

    def open_history_menu(self):
        if self.editor_history is None:return
        self.history_expanded=not self.history_expanded
        self.history_button.configure(text='历史 ▴' if self.history_expanded else '历史 ▾')
        if not self.history_expanded:self.history_panel.grid_remove();return
        if self.history_tree is None or not self.history_tree.winfo_exists():
            from ui_navigation import SlimScroll
            tree=ttk.Treeview(self.history_panel,columns=('time',),show='tree headings',selectmode='browse',style='Editor.Treeview',height=7)
            tree.heading('#0',text='步骤');tree.heading('time',text='时间')
            tree.column('#0',width=185,minwidth=110);tree.column('time',width=72,minwidth=65,stretch=False)
            scroll=SlimScroll(self.history_panel,tree.yview);scroll.pack(side='right',fill='y')
            tree.configure(yscrollcommand=scroll.set);tree.pack(fill='both',expand=True)
            self.history_tree=tree
            tree.bind('<<TreeviewSelect>>',self.editor_history_selected)
            tree.bind('<MouseWheel>',lambda event:(tree.yview_scroll(-int(event.delta/120 or (1 if event.delta>0 else -1))*3,'units'),'break')[-1])
            # Keep this bounded list's wheel handler before the shared panel
            # router even when install_wheel later discovers these widgets.
            for widget in (tree,scroll):
                tags=widget.bindtags();widget.bindtags((tags[0],'PhotoWheel',*tags[1:]))
        self.history_panel.grid();self.history_panel.lift()
        self.refresh_history_tree(force=True);self.editor_sync_toolbar()

    def refresh_history_tree(self,force=False):
        tree = self.history_tree
        if tree is None or not tree.winfo_exists():return
        if self.editor_history is None:
            tree.delete(*tree.get_children());self.history_items={};return
        now=time.monotonic()
        if not force and now-getattr(self,'_history_painted_at',0)<.15:
            if getattr(self,'_history_refresh_after',None) is None:
                def paint():
                    self._history_refresh_after=None;self.refresh_history_tree(force=True)
                self._history_refresh_after=self.root.after(150,paint)
            return
        self._history_painted_at=now
        tree.delete(*tree.get_children()); self.history_items = {}
        for group in self.editor_history.groups(include_states=False):
            for step in group.get('steps', ()):
                sid = 'step:' + str(step['id'])
                tree.insert('', 'end', iid=sid, text=step.get('title', '调整'), values=(_clock(step.get('time')),))
                self.history_items[sid] = step['id']
        current='step:'+str(self.editor_history.current_id)
        if tree.exists(current):tree.selection_set(current);tree.see(current)

    def editor_history_selected(self, event=None):
        if self.editor_loading or self.editor_history is None:return
        selection = self.history_tree.selection()
        if not selection:return
        step = self.history_items.get(selection[0])
        if step is None or step==self.editor_history.current_id:return
        self.editor_history.preview(step); self.apply_editor_state(self.editor_history.current, history_preview=True)
        self.editor_sync_toolbar()

    def open_task_menu(self):
        popup = self._editor_popup('task_popup', '后台任务', self.task_button, width=650, height=310)
        if popup is None:return
        tree = ttk.Treeview(popup, columns=('status',), show='tree headings', selectmode='browse',style='Editor.Treeview',height=6)
        tree.heading('#0', text='照片'); tree.heading('status', text='状态')
        tree.column('#0', width=230); tree.column('status', width=395)
        self.task_tree = tree
        footer=ttk.Frame(popup);footer.pack(side='bottom',fill='x')
        self.task_detail_text=tk.StringVar(master=popup,value='')
        ttk.Label(footer,textvariable=self.task_detail_text,wraplength=620,justify='left',style='Hint.TLabel').pack(fill='x',padx=8,pady=4)
        tree.bind('<<TreeviewSelect>>',self.editor_task_selected)
        self.task_cancel_button=ttk.Button(footer, text='取消选中任务', command=self.editor_cancel_task)
        self.task_cancel_button.pack(anchor='e', padx=4, pady=4)
        tree.pack(fill='both', expand=True, padx=4)
        self.poll_editor_tasks(force=True);self._show_editor_popup(popup)

    def editor_cancel_task(self):
        if self.task_tree is None:return
        selected = self.task_tree.selection()
        if selected:
            if selected[0]=='ai-prewarm':
                import startup_prewarm
                startup_prewarm.stop()
            else:self.photo_denoise_store.cancel_task(selected[0])
            self.poll_editor_tasks(force=True)

    def editor_task_selected(self,event=None):
        if self.task_tree is None or not self.task_tree.winfo_exists():return
        selected=self.task_tree.selection()
        self.task_detail_text.set(self._task_descriptions.get(selected[0],'') if selected else '')

    def poll_editor_tasks(self, force=False):
        if not self.editor_enabled:return
        now=time.monotonic()
        if not force and now-getattr(self,'_task_polled_at',0)<.25:return
        self._task_polled_at=now
        snapshot = self.photo_denoise_store.tasks_snapshot(limit=16)
        import startup_prewarm
        warming=startup_prewarm.snapshot()
        if warming.get('state') in ('queued','running'):
            snapshot=list(snapshot)+[warming]
        active = [row for row in snapshot if row['state'] in ('queued', 'preparing', 'running')]
        indexed={row['task_id']:row for row in snapshot}
        def description(row):
            if row['state'] in ('queued','preparing','running') and row.get('cancel_requested'):
                text='正在停止旧模型' if row.get('model') else '正在停止'
            elif row['state']=='cancelled':text='已被新模型替换' if row.get('cancel_reason')=='superseded' else '已取消'
            elif row['state']=='queued':
                blocked=indexed.get(row.get('blocked_by'))
                if blocked is not None:
                    text='等待 '+blocked['name']+' / '+(blocked.get('model') or '准备照片')
                    if blocked.get('cancel_requested'):text+=' 停止'
                else:text='排队等待'
            else:text=row.get('stage') or row['state']
            percent=row.get('percent')
            if percent is not None and '%' not in text:text+=f' {percent:.0f}%'
            elapsed=row.get('elapsed',0);text+=f' · 用时 {int(elapsed)}秒'
            if row['state'] in ('queued','preparing','running'):
                eta=row.get('wait_eta_seconds') if row['state']=='queued' else row.get('eta_seconds')
                text+=' · 剩余估算中' if eta is None else f' · 剩余约 {max(1,round(eta))}秒'
            if row.get('protected_consumers'):text+=' · 导出使用中'
            return text
        for i,value in enumerate(self.task_summaries):
            row=active[i] if i<len(active) else None
            text='' if row is None else row['name']+' · '+(row.get('model') or '准备照片')+'\n'+description(row)
            if value.get()!=text:value.set(text)
        summary = f'后台任务 {len(active)}' if active else '后台任务'
        if self.task_text.get() != summary:self.task_text.set(summary)
        tree = self.task_tree
        if tree is None or not tree.winfo_exists():return
        stamp = json.dumps(snapshot, sort_keys=True, default=str)
        if not force and stamp == self._task_stamp:return
        self._task_stamp = stamp
        selected = tree.selection(); tree.delete(*tree.get_children())
        self._task_descriptions={row['task_id']:row['name']+' · '+(row.get('model') or '准备照片')+'\n'+description(row) for row in snapshot}
        for row in snapshot:
            tree.insert('', 'end', iid=row['task_id'], text=row['name']+' · '+(row.get('model') or '准备照片'), values=(description(row),))
        if selected and tree.exists(selected[0]):tree.selection_set(selected[0])
        elif snapshot:tree.selection_set(snapshot[0]['task_id'])
        self.editor_task_selected()
