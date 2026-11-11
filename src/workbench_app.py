"""3.1 workbench layout, with original-engine photo processing."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json, math, os, queue, re, sys, threading, time, traceback
from ui_language import (
    tk, ttk, messagebox, simpledialog, filedialog, OPTIONS as LANGUAGE_OPTIONS,
    current_language, set_language, source_caption, tr,
)
from PIL import Image, ImageTk
import numpy as np
import app as classic
from app_paths import CONFIG, report_dir
from native_backend import LOOKS, SEASONS, RenderSession, RenderCancelled, save, mix_amount
from native_controls import available,validate_overrides,NR_CONTROLS
from native_app import preview, App as NativeDrawing
from photo_formats import FORMAT_CHOICES, destination_for
from filter_catalog import (PENTAX, RICOH, CUSTOM, BRANDS, GR_UNAVAILABLE, catalog,
                            groups, PENTAX_GROUPS, require_renderer, CREATIVE_LOOKS, PENTAX_LOOKS,
                            GR_AVAILABLE, GR_MONO, GR_ENGLISH, GR_NOTICE, renderer_available)
from filter_export import export_filters
from legacy_effects import CreativePreview,CreativeSession,save as save_creative
from gr4_backend import GRSession,save as save_gr4
from gr4_renderer import validate_controls as validate_gr4,default_tone,tone_count
from color_balance import BalancePad,apply_image,validate as validate_balance
from tone_tools import TonePanel,histogram
from postprocess import validate as validate_post,preview as post_preview,blend_clean
import gpu_compute
from ui_navigation import SlimScroll,PhotoViewport
from native_noise_panel import NativeNoisePanel
from denoise_detail import DenoiseDetail
from ui_controls import FineSlider,CollapsibleSection,fold_existing,Tooltip
from adjustment_panels import BasicPanel,ColorMixerPanel
from source_denoise_ui import SourceDenoiseMixin
from preview_adjustments import PreviewAdjustmentsMixin
from custom_lut_ui import CustomLutMixin
from preset_thumbnails import PresetThumbnailMixin
from editor_ui import EditorMixin,EditorNativeNoisePanel,DENOISE_CHOICES

VERSION='3.5.13'
classic.VERSION=VERSION
classic.PRESETS=CONFIG/'native-presets.json'
GROUPS=PENTAX_GROUPS
LABELS={'saturation':'饱和度','hue':'色相','key':'影调','contrast':'对比度',
        'highlight':'高光对比度','shadow':'阴影对比度','sharpness':'锐度',
        'filter_effect':'黑白滤镜效应','mono_toning':'黑白色调','toning':'色调'}
FILTER_COLOURS=('本色','绿色','黄色','橙色','红色','洋红色','蓝色','青绿色','红外线')
TONING_COLOURS=('本色','绿色','黄色','橙色','红色','洋红色','紫色','蓝色','青绿色')
ORDER=['saturation','hue','key','contrast','highlight','shadow','sharpness','filter_effect','mono_toning','toning']
BG='#1c1c1e';PANEL='#2c2c2e';FG='#f2f2f7';MUTED='#aeaeb2';BLUE='#0a84ff';EDGE='#48484a'
DENOISE_MODELS={'dcu':'DCU NoiseCut','nafnet':'NAFNet','restormer':'Restormer'}
DENOISE_PREPARE_TIMEOUT=180.


def widgets(parent):
    for child in parent.winfo_children():
        yield child
        yield from widgets(child)


def valid_preset(value):
    if not isinstance(value,dict):return False
    brand=value.get('brand',PENTAX);key=value.get('look')
    if not renderer_available(brand,key):return False
    creative=brand==PENTAX and key in CREATIVE_LOOKS
    ricoh=brand==RICOH
    custom=brand==CUSTOM
    if creative and value.get('pipeline')!='legacy31':return False
    if ricoh and value.get('pipeline') not in ('gr4_r2','gr4_r3'):return False
    if custom and value.get('pipeline')!='custom_lut':return False
    if not creative and not ricoh and not custom and value.get('native') is not True:return False
    try:
        validate_balance(value.get('post_balance',[0,0]))
        validate_post(value.get('post_settings'))
        for field,lo,hi in [('amount',0,200),('exposure',-3,3)]:
            x=value[field]
            if type(x) not in (float,int) or not math.isfinite(x) or not lo<=x<=hi:return False
        if ricoh:validate_gr4(key,value.get('controls',{}))
        elif creative or custom:
            if value.get('controls',{}):return False
        else:validate_overrides(LOOKS[value['look']]['mode'],value.get('controls',{}))
    except (KeyError,TypeError,ValueError):return False
    return True


def _is_baseline_signature(sig):
    """AUDIT-FIX P14: the (0.0,(),1.0) constant was written for the legacy 4-tuple
    signature. The editor path adds a fifth 'stamp' element, so comparing sig[1:]
    against a 3-tuple could never be true and the baseline exclusion silently
    stopped working. Compare only the middle (exposure, controls, amount) triple."""
    if not isinstance(sig, tuple) or len(sig) < 4:
        return False
    return sig[1:4] == (0.0, (), 1.0)


class App(EditorMixin,PresetThumbnailMixin,CustomLutMixin,SourceDenoiseMixin,PreviewAdjustmentsMixin,classic.App):
    def __init__(self,root):
        self.brand=tk.StringVar(master=root,value=PENTAX);self.brand_states={};self.brand_buttons={}
        self.init_custom_lut(root)
        self.filter_dialog=None;self.filter_format_vars={}
        self.closed=False;self.ready=False;self.overrides={};self.factory={};self.control_widgets={}
        self.brand_overrides={}
        self.native_cache={};self.requested=None;self.request_revision=0;self.preview_order=GROUPS['彩色']
        self.preview_error='';self.preview_started=0;self.preview_stage='';self.preview_stats={};self.preview_times={}
        self.path=None;self.full_cache=None;self.full_lock=threading.Lock();self.export_thread=None
        self.active_preset=None
        self.balance_values={};self.balance_pad=None;self.balance_pending=None
        self.post_values={};self.tone_panel=None;self.basic_panel=None;self.mixer_panel=None;self.sections={};self.post_revision=0;self.post_future=None;self.post_request=None
        self.post_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='PhotoAdjust')
        self.card_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='PresetPreview')
        self.post_cache=None;self.post_completed=0;self.post_running=None
        self.input_histogram=None;self.output_histogram=None;self.histogram_key=None
        self.ai_device_rows=[];self.ai_device_order_keys=[];self.ai_device_saved_keys=[];self.ai_device_drag_index=None
        self._background_preparation_after=None;self._background_idle_at=0.
        self._color_warmup_thread=None;self._prewarm_attempted=False
        self.want_detail=False;self.detail_cache=None;self.viewport=None
        self.denoise_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='FullPhotoDenoise')
        self.denoise_future=None;self.denoise_waiting=None;self.denoise_job=None
        self.denoise_serial=0;self.denoise_result=None;self.denoise_error=''
        self.denoise_prepare_started=0.;self.denoise_prepare_text='';self.denoise_detail=None
        self.init_source_denoise()
        self.init_preset_thumbnails()
        self.init_editor_state()
        self.cancel_export=threading.Event();self.export_cancelled=False
        self.cache_evictions=queue.SimpleQueue()
        self.worker_failed=False
        previous_callback=root.report_callback_exception
        self.callback_observer=previous_callback if getattr(previous_callback,'__func__',None) is not tk.Tk.report_callback_exception else None
        self.callback_error_count=0;self.last_callback_error=''
        root.report_callback_exception=self.report_callback_exception
        super().__init__(root)
        # The legacy constructor only knows its legacy look names. Read the
        # separate native preset file again so seasonal presets survive restart.
        try:saved=json.loads(classic.PRESETS.read_text(encoding='utf-8'))
        except (OSError,ValueError):saved={}
        if not isinstance(saved,dict):saved={}
        self.user_presets={k:v for k,v in saved.items() if valid_preset(v)}
        self.look.set('')
        self.group.set('全部')
        self.group_combo.configure(values=[*GROUPS,'我的预设'])
        for w in list(widgets(root)):
            if isinstance(w,ttk.Label):
                text=str(source_caption(w))
                if text=='预设 · 缩略图为标准效果':
                    parent=w.master;w.destroy()
                    bar=ttk.Frame(parent);bar.grid(row=2,column=0,sticky='ew')
                    bar.columnconfigure(tuple(range(len(BRANDS))),weight=1,uniform='brand')
                    for column,(brand,label) in enumerate(BRANDS.items()):
                        button=ttk.Button(bar,text=label,command=lambda b=brand:self.switch_brand(b))
                        button.grid(row=0,column=column,sticky='ew',padx=(0,3) if column<len(BRANDS)-1 else 0)
                        self.brand_buttons[brand]=button
                    self.build_custom_controls(bar)
                elif text.startswith('已内置视频编码器'):
                    w.destroy()
            if isinstance(w,ttk.Button):
                text=str(source_caption(w))
                if text=='导出 LUT / 配置':w.configure(text='导出滤镜文件');self.filter_export_button=w
                if text=='安装 Lightroom 配置':w.destroy()
        right=self.format_combo.master
        first=right.winfo_children()[0]
        original_right=list(right.pack_slaves())
        export_index=next(i for i,w in enumerate(original_right) if isinstance(w,ttk.Label) and source_caption(w)=='导出')
        original_adjust=original_right[:export_index];original_export=original_right[export_index:]
        self.tone_panel=TonePanel(right,self.tone_changed);self.tone_panel.pack(fill='x',pady=(0,10),before=first)
        acceleration=ttk.Frame(right);acceleration.pack(fill='x',pady=(0,10),before=first)
        self.acceleration=tk.StringVar(value='自动')
        ttk.Label(acceleration,text='色彩预览加速').pack(anchor='w',pady=(0,2))
        self.acceleration_combo=ttk.Combobox(acceleration,textvariable=self.acceleration,values=('自动','仅 CPU'),state='readonly')
        self.acceleration_combo.pack(fill='x');self.acceleration_combo.bind('<<ComboboxSelected>>',self.acceleration_changed)
        self.acceleration_text=tk.StringVar(value='正在检测可用加速方式…')
        ttk.Label(acceleration,textvariable=self.acceleration_text,wraplength=258,style='Hint.TLabel').pack(fill='x')
        ttk.Label(acceleration,text='AI 降噪设备顺序').pack(anchor='w',pady=(8,2))
        self.ai_device_order=tk.Listbox(acceleration,height=3,exportselection=False,activestyle='none',
                                        relief='flat',borderwidth=0,highlightthickness=1,
                                        highlightbackground=EDGE,highlightcolor=BLUE)
        self.ai_device_order.pack(fill='x')
        self.ai_device_error=''
        self.ai_device_hint=tk.StringVar(value=tr('正在检测显卡…'))
        ttk.Label(acceleration,textvariable=self.ai_device_hint,wraplength=258,style='Hint.TLabel').pack(fill='x',pady=(3,0))
        self.ai_device_order.bind('<Button-1>',self.ai_device_drag_start,add='+')
        self.ai_device_order.bind('<B1-Motion>',self.ai_device_drag,add='+')
        self.ai_device_order.bind('<ButtonRelease-1>',self.ai_device_drag_end,add='+')
        denoise_box=ttk.LabelFrame(right,text='降噪',padding=8);denoise_box.pack(fill='x',pady=(0,10),before=first)
        self.denoise_enabled=tk.BooleanVar(value=False)
        self.denoise_model=tk.StringVar(value=DENOISE_CHOICES['off'])
        self.denoise_combo=ttk.Combobox(denoise_box,textvariable=self.denoise_model,values=tuple(DENOISE_CHOICES.values()),state='readonly')
        self.denoise_combo.pack(fill='x');self.denoise_combo.bind('<<ComboboxSelected>>',self.denoise_changed)
        self.native_noise_panel=EditorNativeNoisePanel(self,denoise_box)
        self.denoise_amount=tk.DoubleVar(value=100);self.denoise_label=tk.StringVar(value='关闭')
        ttk.Label(denoise_box,textvariable=self.denoise_label,style='Hint.TLabel').pack(fill='x')
        mixture=CollapsibleSection(denoise_box,'原图混合',expanded=False);mixture.pack(fill='x')
        ttk.Label(mixture.body,text='降噪结果占比',style='Hint.TLabel').pack(anchor='w')
        FineSlider(mixture.body,variable=self.denoise_amount,from_=0,to=100,resolution=1,default=100,command=self.denoise_changed).pack(fill='x')
        Tooltip(mixture.header,'降噪结果的混合比例。100% 使用完整结果；不改变模型强度。')
        self.denoise_detail=DenoiseDetail(self,denoise_box)
        self.denoise_start_button=ttk.Button(denoise_box,text='开始全尺寸降噪',command=self.start_denoise)
        self.denoise_start_button.pack(fill='x',pady=(5,3))
        result_actions=ttk.Frame(denoise_box);result_actions.pack(fill='x',pady=(0,3))
        self.denoise_regenerate_button=ttk.Button(result_actions,text='重新生成降噪',command=self.regenerate_denoise,state='disabled')
        self.denoise_regenerate_button.pack(side='left',fill='x',expand=True,padx=(0,3))
        self.denoise_discard_button=ttk.Button(result_actions,text='丢弃降噪结果',command=self.discard_denoise,state='disabled')
        self.denoise_discard_button.pack(side='left',fill='x',expand=True)
        self.denoise_cancel_button=ttk.Button(denoise_box,text='取消降噪',command=self.cancel_denoise,state='disabled')
        self.denoise_cancel_button.pack(fill='x',pady=(0,5))
        progress_row=ttk.Frame(denoise_box);progress_row.pack(fill='x')
        self.denoise_progress=ttk.Progressbar(progress_row,maximum=100,mode='determinate');self.denoise_progress.pack(side='left',fill='x',expand=True)
        self.denoise_progress_text=tk.StringVar(value='—');ttk.Label(progress_row,textvariable=self.denoise_progress_text,width=5,anchor='e').pack(side='right')
        self.denoise_state=tk.StringVar(value='')
        ttk.Label(denoise_box,textvariable=self.denoise_state,wraplength=250,style='Hint.TLabel').pack(fill='x',pady=(4,0))
        Tooltip(self.denoise_combo,'AI 在输入照片上降噪并复用结果；DCU 使用宾得原厂显影链。')
        export_title=next(w for w in right.winfo_children() if isinstance(w,ttk.Label) and source_caption(w)=='导出')
        self.controls_frame=ttk.LabelFrame(right,text='当前滤镜调整',padding=8)
        self.controls_frame.pack(fill='x',pady=(0,12),before=export_title)
        self.controls_message=tk.StringVar()
        self.native_control_body=ttk.Frame(self.controls_frame);self.native_control_body.pack(fill='x')
        ttk.Label(self.controls_frame,textvariable=self.controls_message,wraplength=250,style='Hint.TLabel').pack(fill='x',pady=(5,0))
        self.reset_controls_button=ttk.Button(self.controls_frame,text='恢复此滤镜原厂设置',command=self.reset_controls)
        self.reset_controls_button.pack(fill='x',pady=(6,0))
        balance_frame=ttk.LabelFrame(right,text='色彩平衡',padding=8)
        balance_frame.pack(fill='x',pady=(0,12),before=export_title)
        self.balance_pad=BalancePad(balance_frame,self.balance_changed);self.balance_pad.pack(fill='x')
        self.cancel_button=ttk.Button(right,text='取消导出',command=self.cancel_export.set,state='disabled')
        self.cancel_button.pack(fill='x',before=self.progress,pady=(0,5))
        self.build_adjustment_layout(right,original_adjust,original_export,acceleration,denoise_box,balance_frame)
        self.build_editor_toolbar(right)
        self.build_language_menu()
        self.photo_adjust_widgets=[w for w in widgets(right) if isinstance(w,(ttk.Scale,FineSlider)) or
            isinstance(w,ttk.Button) and source_caption(w) in ('不加','轻柔','标准','加强','重置调色','保存我的预设')]
        views=self.canvas.master.winfo_children()[0]
        ttk.Button(views,text='分界线居中',command=self.reset_split).pack(side='right')
        self.zoom_text=tk.StringVar(value='适应窗口');self.viewport=PhotoViewport(self)
        ttk.Button(views,text='100%',width=5,command=self.viewport.actual).pack(side='right',padx=3)
        ttk.Button(views,text='适应',width=5,command=self.viewport.fit).pack(side='right',padx=3)
        ttk.Label(views,textvariable=self.zoom_text,style='Hint.TLabel').pack(side='right',padx=4)
        self.canvas.bind('<Motion>',self.viewport.pointer);self.canvas.bind('<Button-1>',self.viewport.down)
        self.canvas.bind('<B1-Motion>',self.viewport.drag);self.canvas.bind('<ButtonRelease-1>',lambda e:setattr(self.viewport,'dragging',None))
        self.canvas.bind('<Double-Button-1>',lambda e:self.viewport.fit())
        self.right_canvas=right.master;self.nav_scrollbars=[]
        self.left_scroll_host=self.file_list.master;self.right_scroll_host=self.right_canvas.master
        for canvas in (self.preset_canvas,self.right_canvas):
            for child in canvas.master.winfo_children():
                if isinstance(child,ttk.Scrollbar):child.destroy()
            bar=SlimScroll(canvas.master,canvas.yview);bar.pack(side='right',fill='y');canvas.configure(yscrollcommand=bar.set);self.nav_scrollbars.append(bar)
        self.root.bind_class('PhotoWheel','<MouseWheel>',self.route_wheel)
        self.root.title(classic.APP_NAME+' '+VERSION)
        self.ready=True;self.apply_dark();self.update_brand_controls();self.build_cards();self.build_controls();self.settings_changed();self.install_wheel()
        threading.Thread(target=self.load_ai_device_order,name='AIDeviceDiscovery',daemon=True).start()
        root.protocol('WM_DELETE_WINDOW',self.request_close)

    def build_language_menu(self):
        labels = {code: name for name, code in LANGUAGE_OPTIONS.items()}
        self.language_choice = tk.StringVar(master=self.root, value=labels[current_language()])
        frame = ttk.Frame(self.header)
        frame.pack(side='right', padx=(0, 8))
        ttk.Label(frame, text='语言 / Language', style='Hint.TLabel').pack(side='left', padx=(0, 4))
        self.language_menu = ttk.Combobox(
            frame, textvariable=self.language_choice, values=tuple(LANGUAGE_OPTIONS),
            translate_choices=False, state='readonly', width=8,
        )
        self.language_menu.pack(side='left')
        self.language_menu.bind('<<ComboboxSelected>>', self.change_language)

    def change_language(self, _event=None):
        try:
            set_language(LANGUAGE_OPTIONS[self.language_choice.get()])
        except OSError:
            self.language_choice.set(next(name for name, code in LANGUAGE_OPTIONS.items()
                                          if code == current_language()))
            messagebox.showerror('语言', '无法保存语言设置。', parent=self.root)
            return
        if self.history_expanded:
            self.refresh_history_tree(force=True)
        self.poll_editor_tasks(force=True)
        self.update_ai_device_hint()
        self.render_ai_device_order()

    def load_ai_device_order(self):
        try:
            from neural_devices import priority_rows
            rows=priority_rows()
            error=''
        except Exception as exc:
            rows=[];error=type(exc).__name__
        def apply():
            if self.closed:return
            self.ai_device_rows=rows
            self.ai_device_order_keys=[row['key'] for row in rows]
            self.ai_device_saved_keys=list(self.ai_device_order_keys)
            self.ai_device_error=error
            self.ai_device_order.configure(height=max(2,min(6,len(rows))))
            self.update_ai_device_hint()
            self.render_ai_device_order()
            self._warn_unsupported_graphics(rows)
        try:self.root.after(0,apply)
        except tk.TclError:pass

    def _warn_unsupported_graphics(self, rows):
        # AUDIT-P39: DirectML needs a D3D12 device. Without one the AI models run on
        # CPU, roughly ten times slower, and the only visible sign is a status line.
        # Warn once per launch so the reason is explicit and dismissible.
        if getattr(self, '_graphics_warned', False):return
        from neural_devices import unsupported_graphics
        names = list(dict.fromkeys(unsupported_graphics(rows)))
        if not names:return
        self._graphics_warned = True
        messagebox.showwarning(
            '显卡不支持 AI 降噪',
            '检测到以下显卡不支持 DirectX 12：\n\n    ' + '\n    '.join(names) + '\n\n'
            'AI 降噪（NAFNet / Restormer）通过 DirectML 运行，必须由支持 DirectX 12 的显卡'
            '提供算力。典型门槛是 NVIDIA GTX 900（Maxwell）及更新型号；'
            'GTX 700 / 600 仅部分支持，GTX 500 及更早的 Fermi 架构完全不支持。\n\n'
            '因此降噪会自动改用 CPU，速度约为显卡的十分之一。\n'
            '如果只需要降噪而不追求速度，可以直接使用「DCU 原厂降噪」，它不经过 DirectML。')

    def _schedule_background_preparation(self, delay=400):
        if self.closed:return
        if self._background_preparation_after is not None:
            self.root.after_cancel(self._background_preparation_after)
        self._background_preparation_after=self.root.after(delay,self.start_background_preparation)

    def defer_background_preparation(self):
        if self.closed or not self.ready:return
        self._background_idle_at=time.monotonic()+1.5
        import startup_prewarm
        if startup_prewarm.snapshot().get('state') in ('queued','running'):
            startup_prewarm.stop();self._prewarm_attempted=False
        self._schedule_background_preparation(1600)

    def start_background_preparation(self):
        self._background_preparation_after=None
        if self.closed:return
        if (time.monotonic()<self._background_idle_at or self.busy or self.preview_started
                or not self.post_idle() or not self.denoise_idle()):
            self._schedule_background_preparation();return
        if (gpu_compute.status()['mode']=='auto' and gpu_compute.status()['state']=='waiting'
                and os.environ.get('PHOTO_GPU_DISABLE')!='1'
                and (self._color_warmup_thread is None or not self._color_warmup_thread.is_alive())
                ):
            self._color_warmup_thread=gpu_compute.start_background_warmup()
        if (self._color_warmup_thread is not None and self._color_warmup_thread.is_alive()) or gpu_compute.status()['state']=='checking':
            self._schedule_background_preparation();return
        if self._prewarm_attempted or os.environ.get('PHOTO_PREWARM_DISABLE')=='1':return
        import startup_prewarm
        if startup_prewarm.start(delay=0.):self._prewarm_attempted=True
        else:self._schedule_background_preparation()

    def update_ai_device_hint(self):
        if not hasattr(self,'ai_device_hint'):return
        value=tr('设备检测失败：{0}').format(self.ai_device_error) if self.ai_device_error else tr('拖动以调整 AI 降噪优先级')
        self.ai_device_hint.set(value)

    def render_ai_device_order(self):
        if not hasattr(self,'ai_device_order'):return
        by_key={row['key']:row for row in self.ai_device_rows}
        self.ai_device_order.delete(0,'end')
        for key in self.ai_device_order_keys:
            row=by_key.get(key,{})
            kind={'discrete':tr('独立显卡'),'integrated':tr('集成显卡'),'cpu':'CPU'}.get(row.get('kind'),'GPU')
            name='CPU' if key=='cpu' else row.get('name','GPU')
            self.ai_device_order.insert('end',f'{kind}  ·  {name}')

    def ai_device_drag_start(self,event):
        index=self.ai_device_order.nearest(event.y)
        self.ai_device_drag_index=index if 0<=index<len(self.ai_device_order_keys) else None

    def ai_device_drag(self,event):
        source=self.ai_device_drag_index
        if source is None or not self.ai_device_order_keys:return 'break'
        target=max(0,min(len(self.ai_device_order_keys)-1,self.ai_device_order.nearest(event.y)))
        if source!=target:
            key=self.ai_device_order_keys.pop(source);self.ai_device_order_keys.insert(target,key)
            self.ai_device_drag_index=target;self.render_ai_device_order()
            self.ai_device_order.selection_set(target);self.ai_device_order.activate(target)
        return 'break'

    def ai_device_drag_end(self,event=None):
        self.ai_device_drag_index=None
        if self.ai_device_order_keys==self.ai_device_saved_keys:return
        try:
            from neural_devices import save_priority
            save_priority(self.ai_device_order_keys)
        except Exception:
            self.ai_device_order_keys=list(self.ai_device_saved_keys)
            self.render_ai_device_order()
            messagebox.showerror('AI 设备顺序','设备列表已变化或无法保存，请重新打开设置。',parent=self.root)
            return
        self.ai_device_saved_keys=list(self.ai_device_order_keys)
        self.restart_ai_denoise_for_devices()

    def restart_ai_denoise_for_devices(self):
        # AUDIT-P24: a new device order invalidates the running AI job. PhotoKey is
        # (photo, model, backend_mode) and does not contain the device order, so the
        # task already running on the old device would otherwise be reused.
        if not getattr(self, "current_path", None):return
        choice = (self.editor_state or {}).get("denoise") or {}
        if choice.get("kind") != "ai" or float(choice.get("amount") or 0) <= 0:return
        try:
            key = self.denoise_key()
            if key is not None:self.photo_denoise_store.cancel(key)
        except Exception:pass
        self.denoise_error = ""
        self.status.set("AI 设备顺序已更新，正在用新设备重新开始降噪。")
        self.editor_history_only = False
        self._denoise_request = (str(self.current_path), choice["model"])
        try:self.root.after(80, self._restart_ai_denoise_now)
        except tk.TclError:pass

    def _restart_ai_denoise_now(self):
        if self.closed or self.busy or not self.current_path:return
        choice = (self.editor_state or {}).get("denoise") or {}
        if choice.get("kind") != "ai" or float(choice.get("amount") or 0) <= 0:return
        self.start_denoise()

    def build_adjustment_layout(self,right,adjust,export,acceleration,denoise,balance):
        # Keep every original parent and callback. Only geometry containers
        # change, so the left presets and centre viewport stay in place.
        def section(key,title,children,expanded=False):
            pane=CollapsibleSection(right,title,expanded=expanded);self.sections[key]=pane
            for widget in children:
                options=dict(widget.pack_info()) if widget.winfo_manager()=='pack' else {'fill':'x'}
                options.pop('in',None);widget.pack_forget()
                if isinstance(widget,ttk.Separator) or isinstance(widget,ttk.Label) and str(source_caption(widget)) in ('调色','导出'):continue
                pane.adopt(widget,**options)
            return pane
        filter_section=section('filter','滤镜',adjust,True)
        native_section=fold_existing(self.controls_frame,'当前滤镜调整',False);self.sections['native']=native_section
        basic_section=section('basic','基础调整',[],False)
        self.basic_panel=BasicPanel(basic_section.body,self.basic_changed);self.basic_panel.pack(fill='x')
        mixer_section=section('mixer','颜色混合器',[],False)
        self.mixer_panel=ColorMixerPanel(mixer_section.body,self.mixer_changed);self.mixer_panel.pack(fill='x')
        detail_section=section('detail','细节',[],False);self.basic_panel.build_detail(detail_section.body)
        denoise_section=fold_existing(denoise,'降噪',False);self.sections['denoise']=denoise_section
        balance_section=fold_existing(balance,'色彩平衡',False);self.sections['balance']=balance_section
        performance_section=section('performance','性能', [acceleration],False)
        export_section=section('export','导出',[*export[:-1],self.cancel_button,export[-1]],True)
        ordered=(self.tone_panel,filter_section,native_section,basic_section,mixer_section,detail_section,
                 denoise_section,balance_section,performance_section,export_section)
        for widget in ordered:widget.pack_forget()
        for widget in ordered:widget.pack(fill='x',pady=(0,8))

    def basic_changed(self,basic,regions):
        if not self.ready or self.busy:return
        if self.editor_enabled:
            settings=validate_post(self.editor_state['post']);settings['basic']=basic;settings['tone']=dict(settings['tone'],regions=regions)
            return self.editor_manual_changed('基础调整','basic',post=settings)
        settings=self.current_post();settings['basic']=basic;settings['tone']=dict(settings['tone'],regions=regions)
        self.post_values[(self.brand.get(),self.look.get())]=validate_post(settings)
        self.tone_panel.set(settings['tone']);self.schedule_post()

    def mixer_changed(self,mixer):
        if not self.ready or self.busy:return
        if self.editor_enabled:
            settings=validate_post(self.editor_state['post']);settings['mixer']=mixer
            return self.editor_manual_changed('颜色混合器','mixer',post=settings)
        settings=self.current_post();settings['mixer']=mixer
        self.post_values[(self.brand.get(),self.look.get())]=validate_post(settings);self.schedule_post()

    def install_wheel(self):
        if not self.ready:return
        for w in widgets(self.root):
            tags=w.bindtags()
            if 'PhotoWheel' not in tags:w.bindtags(('PhotoWheel',*tags))

    def route_wheel(self,event):
        delta=getattr(event,'delta',0)
        if not delta:return 'break'
        w=event.widget;pointer=False
        # Windows may deliver MouseWheel to the focused widget. Route by the
        # actual hovered region, including blank panel hosts and scroll thumbs.
        if hasattr(event,'x_root') and hasattr(event,'y_root'):
            try:w=self.root.winfo_containing(event.x_root,event.y_root);pointer=True
            except (tk.TclError,KeyError):return
            if w is None:return 'break'
        hovered=w;steps=-int(delta/120 or (1 if delta>0 else -1))*3
        while w is not None:
            if self.denoise_detail is not None and w is self.denoise_detail.canvas:return self.denoise_detail.wheel(event)
            if w is self.canvas:
                wheel=SimpleNamespace(delta=delta,x=event.x_root-self.canvas.winfo_rootx(),y=event.y_root-self.canvas.winfo_rooty()) if pointer else event
                return self.viewport.wheel(wheel)
            if w is self.file_list:self.file_list.yview_scroll(steps,'units');return 'break'
            if w is self.right_canvas or w is self.right_scroll_host:self.right_canvas.yview_scroll(steps,'units');return 'break'
            if w is self.preset_canvas or w is self.left_scroll_host:self.preset_canvas.yview_scroll(steps,'units');return 'break'
            w=getattr(w,'master',None)
        if pointer and hovered is not event.widget:return 'break'

    def update_brand_controls(self):
        enabled=bool(self.current_path) if self.editor_enabled else renderer_available(self.brand.get(),self.look.get())
        for brand,button in self.brand_buttons.items():
            button.configure(style='Accent.TButton' if brand==self.brand.get() else 'TButton',state='disabled' if self.busy or self.custom_importing else 'normal')
        for w in self.photo_adjust_widgets:w.configure(state='normal' if enabled else 'disabled')
        self.reset_controls_button.configure(state='normal' if enabled else 'disabled')
        self.run_button.configure(state='normal' if enabled and not self.busy and not self.custom_importing else 'disabled')
        self.cancel_button.configure(state='normal' if self.busy else 'disabled')
        self.tabs.tab(1,state='normal' if self.brand.get()==PENTAX else 'disabled')
        if self.balance_pad:self.balance_pad.set_enabled(enabled)
        if self.tone_panel:self.tone_panel.set_enabled(enabled and not self.busy)
        if self.basic_panel:self.basic_panel.set_enabled(enabled and not self.busy)
        if self.mixer_panel:self.mixer_panel.set_enabled(enabled and not self.busy)
        self.refresh_denoise_controls()
        self.native_noise_panel.sync()
        self.update_custom_controls()
        if self.editor_enabled:
            for widget in self.photo_adjust_widgets:
                if isinstance(widget,FineSlider) and widget._variable is self.amount:
                    widget.configure(state='normal' if enabled and self.look.get() and self.editor_state['denoise']['kind']!='native' else 'disabled')
            self.editor_sync_toolbar()
        if hasattr(self,'filter_export_button'):
            custom_ready=self.brand.get()!=CUSTOM or bool(self.custom_brand_id and self.custom_library.catalog(brand_id=self.custom_brand_id))
            self.filter_export_button.configure(state='normal' if custom_ready and not self.busy and not self.custom_importing else 'disabled')

    def switch_brand(self,brand,force=False):
        if getattr(self,'editor_enabled',False):return self.editor_switch_brand(brand,force)
        if brand not in BRANDS:raise ValueError('Unknown filter brand')
        if self.busy or self.custom_importing or (self.brand.get()==brand and not force):return
        previous=self.brand.get()
        self.brand_states[previous]=(self.look.get(),self.group.get(),self.amount.get(),self.exposure.get(),self.active_preset)
        self.brand_overrides[previous]=self.overrides
        self.overrides=self.brand_overrides.setdefault(brand,{})
        self.generation+=1;self.request_revision+=1;self.brand.set(brand)
        self.invalidate_denoise();self.detail_cache=None;self.viewport.reset()
        if self.pending:self.root.after_cancel(self.pending);self.pending=None
        self.requested=None;self.before=self.after=None;self.full_cache=None;self.preview_error=''
        self.reset_histogram()
        self.native_cache={};self.factory={};self.preview_times={};self.preview_stats={}
        self.cache_evictions=queue.SimpleQueue();self.worker_failed=False;self.preview_started=0
        choices=self.custom_library.catalog(brand_id=self.custom_brand_id) if brand==CUSTOM and self.custom_brand_id else {} if brand==CUSTOM else catalog(brand)
        look,group,amount,exposure,preset=self.brand_states.get(brand,(next(iter(choices),''),'全部',100,0,None))
        if look not in choices:look=next(iter(choices),'');group='全部';preset=None
        self.look.set(look);self.group.set(group);self.amount.set(amount);self.exposure.set(exposure);self.active_preset=preset
        self.group_combo.configure(values=[*self.filter_groups(brand),'我的预设'],state='readonly')
        self.tabs.select(0);self.update_brand_controls();self.build_cards();self.build_controls();self.settings_changed();self.draw()
        self.preset_canvas.yview_moveto(0)
        if self.current_path:
            self.photo_started=time.monotonic();self.request_render()
            snapshot=self.source_snapshot();self._source_input_token=self.source_token(snapshot)
            self.pool.submit(self.load_worker,self.current_path,self.generation,self.cache_evictions,brand,snapshot)
        else:self.file_info.set('未选择照片');self.status.set('就绪')

    def apply_dark(self):
        self.root.configure(bg=BG);style=ttk.Style(self.root)
        style.configure('.',background=BG,foreground=FG,fieldbackground=PANEL,bordercolor=EDGE,lightcolor=EDGE,darkcolor=EDGE)
        style.configure('TButton',background=PANEL,foreground=FG,borderwidth=0,padding=(9,6))
        style.map('TButton',background=[('active','#3a3a3c'),('disabled','#252527')],foreground=[('disabled','#6e6e73')])
        style.configure('Accent.TButton',background=BLUE,foreground='white')
        style.map('Accent.TButton',background=[('active','#409cff'),('disabled','#3a3a3c')])
        style.configure('Hint.TLabel',foreground=MUTED)
        style.configure('TCombobox',arrowcolor=FG,fieldbackground=PANEL)
        style.map('TCombobox',fieldbackground=[('readonly',PANEL)],foreground=[('readonly',FG)])
        style.configure('TNotebook.Tab',padding=(18,8),background=PANEL)
        style.map('TNotebook.Tab',background=[('selected','#3a3a3c')],foreground=[('selected',FG)])
        style.configure('Horizontal.TScale',troughcolor=PANEL,background=BLUE)
        style.configure('TEntry',insertcolor=FG)
        self.root.option_add('*TCombobox*Listbox.background',PANEL)
        self.root.option_add('*TCombobox*Listbox.foreground',FG)
        self.root.option_add('*TCombobox*Listbox.selectBackground',BLUE)
        for w in widgets(self.root):
            if isinstance(w,tk.Canvas):w.configure(bg=BG)
            elif isinstance(w,tk.Listbox):w.configure(bg=PANEL,fg=FG,selectbackground=BLUE,selectforeground='white',highlightthickness=0)
        self.canvas.configure(bg='#111113')

    def scroll_cards(self,event):
        self.preset_canvas.yview_scroll(-1 if event.delta>0 else 1,'units');return 'break'

    def build_cards(self):
        brand=self.brand.get();looks=catalog(brand)
        keys=self.filter_groups(brand).get(self.group.get(),tuple('user:'+k for k,v in self.user_presets.items() if v.get('brand',PENTAX)==brand
            and (brand!=CUSTOM or looks.get(v.get('look'),{}).get('brand_id')==self.custom_brand_id)))
        keys=tuple(key for key in keys if (self.user_presets[key[5:]]['look'] if key.startswith('user:') else key) in looks)
        self.preview_order=tuple(look for k in keys for look in (self.user_presets[k[5:]]['look'] if k.startswith('user:') else k,) if renderer_available(brand,look))
        layout=(brand,tuple((key,looks[self.user_presets[key[5:]]['look'] if key.startswith('user:') else key]['name']) for key in keys))
        if layout==self._preset_card_layout:
            self.bind_preset_cards();self.highlight();return
        for w in self.card_frame.winfo_children():w.destroy()
        self.cards={};self._preset_card_layout=layout
        for i,key in enumerate(keys):
            look=self.user_presets[key[5:]]['look'] if key.startswith('user:') else key
            title=key[5:] if key.startswith('user:') else looks[look]['name'] if brand==CUSTOM else GR_ENGLISH.get(look,look) if brand==RICOH else look
            frame=tk.Frame(self.card_frame,bg=PANEL,highlightthickness=2,highlightbackground=EDGE)
            frame.grid(row=i//2,column=i%2,padx=3,pady=4)
            label=tk.Label(frame,text=('尚未开放' if not renderer_available(brand,look) else '等待照片' if not self.current_path else '等待生成'),width=15,height=5,bg=PANEL,fg=MUTED)
            label.pack()
            caption=title if brand==CUSTOM else title+'\n'+looks[look]['name']
            tk.Label(frame,text=caption,font=('Microsoft YaHei UI',9),wraplength=112,width=15,bg=PANEL,fg=FG).pack(pady=5)
            for w in (frame,*frame.winfo_children()):
                w.bind('<Button-1>',lambda e,k=key:self.select_preset(k));w.bind('<MouseWheel>',self.scroll_cards)
            self.cards[key]=(frame,label)
        if not keys:ttk.Label(self.card_frame,text='暂无预设').pack(pady=14)
        self.bind_preset_cards()
        self.highlight()
        if self.ready:self.install_wheel()

    def set_card_image(self,key,im):
        if key not in self.cards:return
        label=self.cards[key][1]
        if getattr(label,'_style_source',None) is im:return
        super().set_card_image(key,im);label._style_source=im

    def highlight(self):
        selected=self.active_preset or self.look.get()
        for key,(frame,_) in self.cards.items():frame.configure(highlightbackground=BLUE if key==selected else EDGE)

    def select_preset(self,key):
        if getattr(self,'editor_enabled',False):return self.editor_filter_selected(key)
        if key.startswith('user:'):
            preset=self.user_presets.get(key[5:])
            if preset and preset.get('brand')==CUSTOM:
                identity=catalog(CUSTOM).get(preset.get('look'),{}).get('brand_id')
                if identity and (self.brand.get()!=CUSTOM or self.custom_brand_id!=identity):self.select_custom_brand(identity)
        # Selecting the standard black-and-white card must not inherit a
        # previous colour filter's partial mix. A saved user preset retains
        # its explicitly chosen amount, including intentional partial colour.
        if key.startswith('Mono') or (self.brand.get()==RICOH and key in GR_MONO):
            self.amount.set(100);self.balance_values[(self.brand.get(),key)]=(0,0)
        self.active_preset=key if key.startswith('user:') else None
        if key.startswith('user:'):
            p=self.user_presets[key[5:]];key=p['look'];self.amount.set(p['amount']);self.exposure.set(p['exposure'])
            controls=dict(p.get('controls',{}))
            noise={k:v for k,v in controls.items() if k in NR_CONTROLS}
            if noise:self.photo_native_noise[self.photo_settings_key()]=noise
            self.overrides[key]={k:v for k,v in controls.items() if k not in NR_CONTROLS}
            self.balance_values[(self.brand.get(),key)]=tuple(p.get('post_balance',[0,0]))
            self.post_values[(self.brand.get(),key)]=validate_post(p.get('post_settings'))
        self.look.set(key);self.highlight()
        wanted=self._preset_requests.get(self.active_preset or key)
        if wanted is not None:self.preset_thumbnails.retry(wanted)
        if self.ready:self.build_controls();self.update_brand_controls()
        self.settings_changed()

    def build_controls(self):
        for w in self.native_control_body.winfo_children():w.destroy()
        self.control_widgets={};key=self.look.get();report=self.factory.get(key)
        self._controls_identity=(self.generation,self.brand.get(),key,bool(report))
        if self.balance_pad:self.balance_pad.set(self.current_balance())
        if self.tone_panel:
            settings=self.current_post();self.tone_panel.set(settings['tone']);self.denoise_amount.set(self.editor_state['denoise']['amount'] if self.editor_enabled else settings['denoise']);self.denoise_label.set('关闭' if not settings['denoise'] else f'结果混合 {settings["denoise"]:.0f}%')
            self.basic_panel.set(settings['basic'],settings['tone']['regions']);self.mixer_panel.set(settings['mixer'])
            if not self.editor_enabled:self.denoise_model.set(DENOISE_MODELS[settings['denoise_model']])
            self.refresh_denoise_controls()
            self.root.after_idle(self.install_wheel)
        if not key:
            self.preset_text.set('原图');self.controls_message.set('');return
        if self.brand.get()==RICOH:
            if key not in GR_AVAILABLE:self.controls_message.set(GR_UNAVAILABLE);return
            from gr4_renderer import control_values
            from gr4_parameters import LABELS as GR_LABELS,ORDER as GR_ORDER
            defaults=control_values(key);current=control_values(key,self.overrides.get(key,{}))
            for field in GR_ORDER:
                if field=='tone' or field not in defaults:continue
                row=ttk.Frame(self.native_control_body);row.pack(fill='x',pady=3)
                ttk.Label(row,text=GR_LABELS[field]).pack(side='left')
                value=tk.IntVar(value=current[field])
                slider=FineSlider(row,from_=-4,to=4,variable=value,width=115,resolution=1,
                    default=defaults[field],command=lambda v,f=field:self.control_changed(f))
                slider.pack(side='right');self.control_widgets[field]=(value,slider)
            if tone_count(key)>1:
                row=ttk.Frame(self.native_control_body);row.pack(fill='x',pady=3)
                ttk.Label(row,text='交叉处理色调' if key=='CrossProcess' else '电影色调').pack(side='left')
                value=tk.IntVar(value=self.overrides.get(key,{}).get('tone',default_tone(key)))
                names=[f'色调 {i}'+('（默认）' if i==default_tone(key) else '') for i in range(tone_count(key))]
                combo=ttk.Combobox(row,values=names,width=12,state='readonly');combo.current(value.get());combo.pack(side='right')
                combo.bind('<<ComboboxSelected>>',lambda e,v=value,w=combo:(v.set(w.current()),self.control_changed('tone')))
                self.control_widgets['tone']=(value,combo)
            self.reset_controls_button.configure(text='恢复此滤镜默认设置')
            self.controls_message.set('')
            return
        if self.brand.get()==CUSTOM:
            self.reset_controls_button.configure(text='恢复此滤镜默认设置')
            self.controls_message.set('CUBE' if renderer_available(CUSTOM,key) else '导入 CUBE 后选择滤镜')
            return
        self.reset_controls_button.configure(text='恢复此滤镜原厂设置')
        if key in CREATIVE_LOOKS:
            self.controls_message.set('无独立参数');return
        if not report:
            self.controls_message.set('等待照片');return
        defaults=report['factory_defaults'];supported=available(LOOKS[key]['mode']);overrides=self.overrides.get(key,{})
        has_hl='highlight' in supported
        self.hl_enabled=tk.BooleanVar(value=report['factory_highlight_shadow_enabled'] or any(x in overrides for x in ('highlight','shadow')))
        for field in ORDER:
            if field not in supported:continue
            if field=='highlight' and has_hl:
                ttk.Checkbutton(self.native_control_body,text='单独调整高光 / 阴影',variable=self.hl_enabled,command=self.toggle_hl).pack(anchor='w',pady=5)
            row=ttk.Frame(self.native_control_body);row.pack(fill='x',pady=3)
            ttk.Label(row,text=LABELS[field]).pack(side='left')
            value=tk.IntVar(value=overrides.get(field,defaults[field]));lo,hi=(0,8) if field in ('toning','filter_effect') else (-4,4)
            if field in ('toning','filter_effect'):
                colours=TONING_COLOURS if field=='toning' else FILTER_COLOURS
                spin=ttk.Combobox(row,values=colours,width=13,state='readonly')
                spin.current(value.get()) if 0<=value.get()<len(colours) else spin.set('原厂默认')
                spin.bind('<<ComboboxSelected>>',lambda e,f=field,v=value,w=spin:(v.set(w.current()),self.control_changed(f)))
                trace=value.trace_add('write',lambda *a,v=value,w=spin:w.current(v.get()) if 0<=v.get()<len(FILTER_COLOURS) else w.set('原厂默认'))
                # Variable traces outlive destroyed widgets unless removed.
                # Release their closures on the Tk thread while the widget is
                # being destroyed, before a background allocation can run GC.
                def release_trace(event,v=value,name=trace,widget=spin):
                    if event.widget is widget:
                        try:v.trace_remove('write',name)
                        except tk.TclError:pass
                spin.bind('<Destroy>',release_trace,add='+')
            else:
                spin=FineSlider(row,from_=lo,to=hi,variable=value,width=115,resolution=1,default=defaults[field],command=lambda v,f=field:self.control_changed(f))
            spin.pack(side='right')
            self.control_widgets[field]=(value,spin)
            if field in ('highlight','shadow') and not self.hl_enabled.get():spin.configure(state='disabled')
        self.controls_message.set('')

    def toggle_hl(self):
        values=self.overrides.setdefault(self.look.get(),{})
        for field in ('highlight','shadow'):
            if field not in self.control_widgets:continue
            var,spin=self.control_widgets[field];spin.configure(state='normal' if self.hl_enabled.get() else 'disabled')
            if self.hl_enabled.get():values[field]=int(var.get())
            else:values.pop(field,None)
        self.settings_changed()

    def control_changed(self,field):
        if self.brand.get()==RICOH:
            from gr4_renderer import control_values
            if field not in self.control_widgets:return
            key=self.look.get();var,widget=self.control_widgets[field]
            values=dict(self.overrides.get(key,{}));previous=control_values(key,values)[field]
            try:
                value=int(var.get());values[field]=value;validate_gr4(key,values)
            except (ValueError,tk.TclError):var.set(previous);return
            if value==previous:return
            self.overrides[key]=values;self.settings_changed();return
        if field not in self.control_widgets or self.look.get() not in self.factory:return
        key=self.look.get();var,spin=self.control_widgets[field]
        if str(spin['state'])=='disabled':return
        lo,hi=(0,8) if field in ('toning','filter_effect') else (-4,4)
        try:value=int(var.get());assert lo<=value<=hi
        except (tk.TclError,ValueError,AssertionError):
            var.set(self.overrides.get(key,{}).get(field,self.factory[key]['factory_defaults'][field]));return
        values=self.overrides.setdefault(key,{})
        if values.get(field,self.factory[key]['factory_defaults'][field])==value:return
        values[field]=value;self.settings_changed()

    def reset_controls(self):
        self.overrides.pop(self.look.get(),None);self.build_controls();self.settings_changed()

    def reset(self):
        if self.editor_enabled:
            from photo_edit import default_state
            value=default_state()
            return self.editor_manual_changed('复位调色','reset',**{key:value[key] for key in ('exposure','balance','post')})
        self.amount.set(100);self.exposure.set(0)
        self.balance_values[(self.brand.get(),self.look.get())]=(0,0)
        self.post_values[(self.brand.get(),self.look.get())]=validate_post()
        if self.ready:self.reset_controls()
        else:self.settings_changed()

    def settings_changed(self):
        if getattr(self,'editor_enabled',False):return self.editor_filter_changed()
        self.amount_text.set(f'浓淡   {self.amount.get():.0f}%')
        if (self.look.get().startswith('Mono') or self.brand.get()==RICOH and self.look.get() in GR_MONO) and round(self.amount.get(),2)!=100:
            self.amount_text.set(f'浓淡   {self.amount.get():.0f}%')
        self.ev_text.set(f'曝光补偿   {self.exposure.get():+.2f} EV')
        key=self.look.get();meta=catalog(self.brand.get()).get(key)
        if meta is None:
            self.preset_text.set('未选择滤镜')
            if self.pending:self.root.after_cancel(self.pending);self.pending=None
            self.requested=None;self.after=None;self.preview_started=0
            if self.ready:self.update_brand_controls();self.draw()
            return
        self.preset_text.set(meta['name'] if self.brand.get()==CUSTOM else key+'\n'+meta['name'])
        self.invalidate_denoise()
        if self.brand.get()==RICOH:
            if key not in GR_AVAILABLE:
                if self.pending:self.root.after_cancel(self.pending)
                self.pending=None;self.requested=None;self.after=None;self.preview_started=0;self.draw();self.status.set(GR_UNAVAILABLE);return
            self.preset_text.set(GR_ENGLISH.get(key,key)+'\n'+catalog(RICOH)[key]['name'])
        if key in CREATIVE_LOOKS:
            self.preset_text.set(key+'\n'+CREATIVE_LOOKS[key]['name'])
        # Fixed-rate scheduling, not a trailing debounce. A continuous drag
        # must actually start work and may show completed intermediate frames.
        if self.pending is None:self.pending=self.root.after(80,self.request_render)

    def signature(self):
        if getattr(self,'editor_enabled',False):return self.editor_signature()
        return (self.look.get(),round(self.exposure.get(),2),tuple(sorted(self.effective_overrides().items())),round(self.amount.get()/100,2))

    def request_render(self):
        if getattr(self,'editor_enabled',False):return self.editor_request_render()
        self.pending=None
        if not renderer_available(self.brand.get(),self.look.get()) or not self.current_path or self.closed:return
        self.defer_background_preparation()
        self.requested=self.signature();self.request_revision+=1;self.preview_error=''
        if self.requested in self.native_cache:self.show_result(*self.native_cache[self.requested]);return
        self.preview_started=time.monotonic();self.preview_stage='正在更新，保留上一帧';self.draw()
        if self.worker_failed:
            self.worker_failed=False;self.pool.submit(self.load_worker,self.current_path,self.generation,self.cache_evictions,self.brand.get(),self.source_snapshot())

    def cache_result(self,token,sig,result):
        self.native_cache.pop(sig,None);self.native_cache[sig]=result
        custom=[s for s in self.native_cache if not _is_baseline_signature(s) and s!=self.requested]
        for old in custom[:-15]:
            del self.native_cache[old];self.cache_evictions.put((token,old))

    def file_selected(self,event=None):
        indices=self.file_list.curselection()
        if not indices:return
        path=self.files[indices[0]]
        if self.editor_enabled:return self.editor_file_selected(path)
        self.defer_background_preparation()
        from photo_denoise import identity_for
        if path==self.current_path and self._preset_photo==identity_for(path):return
        self.generation+=1;self.current_path=self.path=path;self.native_cache={};self.factory={}
        self.preset_photo_selected(path)
        # An explicit preset/control choice made before opening a photo is
        # pending for the first selected input, not an orphan None-key entry.
        for settings in (self.photo_native_noise,self.photo_denoise_values):
            pending=settings.pop(None,None)
            if pending is not None:settings[self.photo_settings_key()]=pending
        self.invalidate_denoise();self.detail_cache=None;self.viewport.reset()
        self.cache_evictions=queue.SimpleQueue()
        self.worker_failed=False
        self.before=self.after=None;self.preview_data=None;self.full_cache=None;self.preview_times={};self.photo_started=time.monotonic();self.preview_error=''
        self.reset_histogram()
        self.requested=self.signature() if renderer_available(self.brand.get(),self.look.get()) else None;self.request_revision+=1;self.preview_started=time.monotonic() if self.requested else 0;self.preview_stage='读取照片'
        self.build_cards();self.build_controls();self.draw();self.export_changed()
        snapshot=self.source_snapshot();self._source_input_token=self.source_token(snapshot)
        self.pool.submit(self.load_worker,path,self.generation,self.cache_evictions,self.brand.get(),snapshot)
        if self.current_photo_denoise()['denoise'] and self.current_denoise() is None:self.start_denoise()

    def clear(self):
        if self.busy:return
        self.generation+=1;self.current_path=self.path=None;self.files=[];self.file_list.delete(0,'end');self.requested=None
        self.preset_photo_selected(None)
        if self.editor_enabled:
            from photo_edit import default_state
            self.editor_history=None;self.editor_state=default_state();self.editor_pending=False;self._editor_requested=None
            self.apply_editor_state(self.editor_state,render=False);self.editor_sync_toolbar()
        self.invalidate_denoise();self.detail_cache=None;self.viewport.reset()
        self.before=self.after=None;self.full_cache=None;self.native_cache={};self.factory={};self.thumbnails={};self.preview_started=0;self.preview_error=''
        self.reset_histogram()
        self.build_cards();self.build_controls();self.draw();self.file_info.set('未选择照片');self.export_changed()
        self.run_button.configure(text='导出列表中的全部照片')

    def load_worker(self,path,token,evictions,brand=PENTAX,snapshot=None):
        if self.editor_enabled:return self.load_editor_worker(path,token)
        snapshot=snapshot or (path,None,0.)
        try:
            with self.source_input(snapshot,lambda:self.closed or token!=self.generation) as working:
                self._load_filter_worker(working,token,evictions,brand)
        except RenderCancelled:pass
        except Exception as exc:self.events.put(('native_error',(token,None,str(exc))))

    # ---------------------------------------------------------------------------
    # LEGACY PATH  (AUDIT-20261111/03-ISSUES-BATCH3.md)
    #
    # Reachable only when App.editor_enabled is False. build_editor_toolbar()
    # sets it to True inside App.__init__, so this path never runs after
    # startup. It is kept as the reference implementation for rollback.
    #
    # Do NOT extend. If you change behaviour here, also confirm the editor path
    # in editor_ui.py / editor_pipeline.py.
    # ---------------------------------------------------------------------------

    def _load_filter_worker(self,path,token,evictions,brand=PENTAX):
        if brand==CUSTOM:return self.load_custom_worker(path,token,evictions)
        if brand==RICOH:return self.load_gr_worker(path,token,evictions)
        try:
            with RenderSession(path) as session:
                creative_session=CreativePreview(path);creative_full=CreativeSession(path)
                completed=set();pending={};failed={};max_pending=0;baseline_previews={};preview_lock=threading.Lock()
                def base_preview(base,alpha):
                    with preview_lock:
                        if id(base) not in baseline_previews:baseline_previews[id(base)]=preview(base,alpha)
                        return baseline_previews[id(base)]
                def task(sig):
                    key,ev,items,amount=sig;native_sig=sig[:3]
                    selected=sig==self.requested
                    def cancelled():
                        if self.closed or token!=self.generation or self.busy:return True
                        if selected:return self.requested is None or key!=self.requested[0]
                        return sig!=self.requested and (self.requested not in completed or not self.denoise_idle())
                    def progress(text):self.events.put(('native_stage',(token,sig,text)))
                    if key in CREATIVE_LOOKS:
                        progress('生成 3.1 创意滤镜预览')
                        base,rgb,alpha,report=creative_session.render(key,amount,ev,cancelled)
                        if self.needs_full_detail() and sig==self.requested:
                            full,full_alpha,full_report=creative_full.render(key,amount,ev,cancelled)
                            self.detail_cache=(token,sig,creative_full.base,full,full_alpha)
                            report.update(native_width=full_report['native_width'],native_height=full_report['native_height'])
                        return classic.to_image(base,alpha),classic.to_image(rgb,alpha),report
                    with self.full_lock:cached=self.full_cache
                    if cached and cached[0]==token and cached[1]==native_sig:
                        _,_,rgb,alpha,report,base=cached
                    else:
                        base,base_alpha=session.baseline(key,progress,cancelled)
                        self.events.put(('native_before',(token,sig,base_preview(base,base_alpha))))
                        rgb,alpha,report=session.render(key,ev,progress,cancelled,dict(items))
                    if cancelled():raise RenderCancelled()
                    if sig==self.requested:
                        with self.full_lock:self.full_cache=(token,native_sig,rgb,alpha,report,base)
                    processed=mix_amount(base,rgb,amount)
                    if sig==self.requested:self.detail_cache=(token,sig,base,processed,alpha)
                    return base_preview(base,alpha),preview(processed,alpha),dict(report,user_amount_percent=amount*100)
                with ThreadPoolExecutor(max_workers=1) as workers:
                    while token==self.generation and not self.closed:
                        try:
                            while True:
                                generation,old=evictions.get_nowait()
                                if generation==token:completed.discard(old)
                        except queue.Empty:pass
                        for future in tuple(pending):
                            if not future.done():continue
                            sig,rev=pending.pop(future)
                            try:
                                result=future.result();completed.add(sig);self.events.put(('native_result',(token,sig,result)))
                                self.events.put(('native_stats',(token,{'decode_count':session.decode_count,'max_parallel':max_pending})))
                            except RenderCancelled:pass
                            except Exception as exc:failed[sig]=rev;self.events.put(('native_error',(token,sig,str(exc))))
                        if not self.busy and self.requested:
                            wanted=self.requested;rev=self.request_revision
                            if self.needs_full_detail() and (self.detail_cache is None or self.detail_cache[:2]!=(token,wanted)):completed.discard(wanted)
                            if wanted in failed and failed[wanted]!=rev:del failed[wanted]
                            active={s for s,_ in pending.values()};limit=1
                            candidates=(wanted,)
                            for sig in dict.fromkeys(candidates):
                                if len(pending)>=limit:break
                                if sig in completed or sig in failed or sig in active:continue
                                pending[workers.submit(task,sig)]=(sig,rev);max_pending=max(max_pending,len(pending))
                        time.sleep(.03)
        except RenderCancelled:pass
        except Exception as exc:self.events.put(('native_error',(token,None,str(exc))))

    def load_gr_worker(self,path,token,evictions):
        """Single decoded input, selected look first, cancel obsolete requests."""
        try:
            session=GRSession(path,preview_only=True,retain_full=True);completed=set();failed={};before=None
            while token==self.generation and not self.closed:
                try:
                    while True:
                        generation,old=evictions.get_nowait()
                        if generation==token:completed.discard(old)
                except queue.Empty:pass
                if self.busy or not self.requested:time.sleep(.03);continue
                wanted=self.requested;revision=self.request_revision
                if self.needs_full_detail() and wanted in completed and (self.detail_cache is None or self.detail_cache[:2]!=(token,wanted)):
                    from gr4_renderer import render as full_render
                    key,ev,items,amount=wanted
                    def detail_cancelled():return self.closed or token!=self.generation or wanted!=self.requested or self.busy or not self.needs_full_detail()
                    try:
                        source=session.full_linear if session.full_linear is not None else session.full_base
                        rgb=full_render(source,key,dict(items),ev,amount,detail_cancelled,input_encoding=session.input_encoding)
                        self.detail_cache=(token,wanted,session.full_base,rgb,None if session.full_alpha is None else session.full_alpha.astype(np.float64)/65535)
                    except RenderCancelled:pass
                    continue
                if wanted in failed and failed[wanted]!=revision:failed.pop(wanted,None)
                candidates=(wanted,)
                sig=next((s for s in dict.fromkeys(candidates) if s not in completed and s not in failed),None)
                if sig is None:time.sleep(.03);continue
                key,ev,items,amount=sig
                selected=sig==self.requested
                def cancelled():
                    return (self.closed or token!=self.generation or self.busy or
                            (selected and (self.requested is None or key!=self.requested[0])) or
                            (not selected and sig!=self.requested and self.requested not in completed))
                def progress(text):self.events.put(('native_stage',(token,sig,text)))
                try:
                    base,alpha=session.prepare(cancelled,progress)
                    display_alpha=None if alpha is None else alpha.astype('float64')/65535
                    if before is None:before=preview(base,display_alpha)
                    self.events.put(('native_before',(token,sig,before)))
                    rgb,_,report=session.process(key,ev,dict(items),amount,cancelled,progress)
                    if cancelled():raise RenderCancelled()
                    result=(before,preview(rgb,display_alpha),report)
                    completed.add(sig);self.events.put(('native_result',(token,sig,result)))
                    self.events.put(('native_stats',(token,{'decode_count':session.decode_count,'max_parallel':1})))
                except RenderCancelled:pass
                except Exception as exc:failed[sig]=revision;self.events.put(('native_error',(token,sig,str(exc))))
        except RenderCancelled:pass
        except Exception as exc:self.events.put(('native_error',(token,None,str(exc))))

    def show_result(self,before,after,report):
        self.before=before;self.preview_started=0;self.preview_error=''
        self.queue_post(after)
        history_hint=' · 此步骤的降噪结果未缓存，需重新计算' if self.editor_enabled and self.editor_history_missing_denoise() else ''
        if report.get('preview_only'):
            self.file_info.set(Path(self.current_path).name+history_hint)
            self.draw();return
        self.file_info.set(f'{Path(self.current_path).name}    {report["native_width"]} × {report["native_height"]}'+history_hint)
        self.draw()

    def current_balance(self):
        if getattr(self,'editor_enabled',False):return tuple(self.editor_state['balance'])
        return self.balance_values.get((self.brand.get(),self.look.get()),(0,0))

    def render_balance(self):
        return (0,0) if self.editor_enabled else self.current_balance()

    def reset_histogram(self):
        self.input_histogram=None;self.output_histogram=None;self.histogram_key=None
        self.refresh_histogram()

    def refresh_histogram(self):
        """Describe the whole preview actually available, even if rendering fails."""
        if not self.ready or self.tone_panel is None:return
        data=None;empty='等待照片'
        if self.before is None:
            if self.preview_error:source='输入未能读取';empty='输入未能读取，暂无统计'
            elif self.current_path:source='输入照片读取中';empty='正在读取输入照片…'
            else:source='尚未选择照片'
        else:
            cached=self.input_histogram
            if cached is None or cached[0]!=self.generation or cached[1] is not self.before:
                cached=(self.generation,self.before,histogram(self.before));self.input_histogram=cached
            data=cached[2];source='输入预览'
            if self.preview_error:source+=' · 处理失败，保留输入统计'
            elif self.after is not None and self.view.get()!='before':
                cached=self.output_histogram
                if cached is None or cached[0]!=self.generation or cached[1] is not self.after:
                    cached=(self.generation,self.after,histogram(self.after));self.output_histogram=cached
                data=cached[2];source='处理后预览'
        key=(self.generation,id(data),source,empty)
        if key!=self.histogram_key:
            self.histogram_key=key;self.tone_panel.set_histogram(data,source,empty)

    def balance_changed(self,value):
        if self.editor_enabled:return self.editor_manual_changed('色彩平衡','balance',balance=list(validate_balance(value)))
        self.balance_values[(self.brand.get(),self.look.get())]=validate_balance(value)
        self.schedule_post(45)

    def refresh_balance(self):
        pending=self.balance_pending;self.balance_pending=None
        if pending:
            try:self.root.after_cancel(pending)
            except tk.TclError:pass
        if self.requested in self.native_cache and self.requested==self.signature():
            self.show_result(*self.native_cache[self.requested])


    def tone_changed(self,value):
        if self.editor_enabled:
            settings=validate_post(self.editor_state['post']);settings['tone']=value
            return self.editor_manual_changed('明暗曲线','tone',post=settings)
        settings=self.current_post();settings['tone']=value;self.post_values[(self.brand.get(),self.look.get())]=settings
        if self.basic_panel:self.basic_panel.set(settings['basic'],value['regions'])
        self.schedule_post()







    def stop_denoise_preparation(self):
        self.denoise_prepare_started=0.;self.denoise_prepare_text=''
        self.denoise_progress.stop();self.denoise_progress.configure(mode='determinate')

    def show_denoise_stage(self,text):
        """Only completed image blocks have a measurable percentage."""
        text=str(text)
        match=re.search(r'(\d+(?:\.\d+)?)%',text)
        percent=float(match.group(1)) if match else None
        measured=percent is not None and math.isfinite(percent) and percent<100
        if measured:
            self.denoise_progress.stop();self.denoise_progress.configure(mode='determinate')
            percent=max(0.,percent)
            self.denoise_progress['value']=percent;self.denoise_progress_text.set(f'{percent:.0f}%')
        else:
            if percent is not None:text='正在准备处理结果…'
            if str(self.denoise_progress['mode'])!='indeterminate':
                self.denoise_progress.configure(mode='indeterminate');self.denoise_progress.start(60)
            self.denoise_progress_text.set('准备')
        if text!=getattr(self,'denoise_stage_text',None):
            self.denoise_stage_text=text;self.denoise_stage_started=time.monotonic()
        self.denoise_stage_measured=measured
        self.denoise_state.set(text)
        if not self.busy:self.status.set(text)

    def fail_denoise_preparation(self,text):
        job=self.denoise_waiting
        if job is not None:job[2].set();self.denoise_waiting=None;self.denoise_serial+=1
        self.stop_denoise_preparation();self.denoise_error=text
        self.denoise_progress['value']=0;self.denoise_progress_text.set('失败')
        self.denoise_state.set(text);self.status.set(text);self.refresh_denoise_controls()







    def acceleration_changed(self,event=None):
        gpu_compute.set_mode('cpu' if self.acceleration.get()=='仅 CPU' else 'auto')
        if self.acceleration.get()!='仅 CPU':self._schedule_background_preparation(0)
        self.schedule_post()





    def draw(self):
        self.refresh_histogram()
        if self.editor_enabled or self.brand.get() in (PENTAX,CUSTOM) or self.look.get() in GR_AVAILABLE:
            return self.viewport.draw(lambda:NativeDrawing.draw(self)) if self.viewport else NativeDrawing.draw(self)
        self.canvas.delete('all');self.canvas.create_text(max(10,self.canvas.winfo_width())/2,max(10,self.canvas.winfo_height())/2,
            text='理光 GR4\n\n此模式尚未开放\n请选择其他可用滤镜',fill=MUTED,width=380,
            font=('Microsoft YaHei UI',12),justify='center')
    drag=NativeDrawing.drag
    reset_split=NativeDrawing.reset_split
    comparison_pointer=NativeDrawing.comparison_pointer

    def save_preset(self):
        if not renderer_available(self.brand.get(),self.look.get()):self.status.set('请先选择滤镜' if self.brand.get()==CUSTOM else GR_UNAVAILABLE);return
        name=simpledialog.askstring('保存我的预设','保存当前滤镜、专属参数和浓淡。照片调色随照片保留。' if self.editor_enabled else '保存当前滤镜、专属调整、浓淡和曝光。',parent=self.root)
        if not name or not name.strip():return
        name=name.strip()[:40]
        if name in self.user_presets and not messagebox.askyesno('同名预设','替换已保存的同名预设？'):return
        brand=self.brand.get();creative=brand==PENTAX and self.look.get() in CREATIVE_LOOKS
        self.user_presets[name]={'brand':brand,'native':brand==PENTAX and not creative,'pipeline':'custom_lut' if brand==CUSTOM else 'gr4_r3' if brand==RICOH else 'legacy31' if creative else 'native','look':self.look.get(),'amount':self.amount.get(),'exposure':self.exposure.get(),'controls':self.effective_overrides(),'post_balance':list(self.current_balance()),'post_settings':dict(self.current_post(),denoise=0)}
        if self.editor_enabled:
            self.user_presets[name].update(exposure=0.,post_balance=[0.,0.],post_settings=validate_post(),
                controls=dict(self.editor_state['filter']['controls']))
        if brand==CUSTOM:self.user_presets[name]['custom_brand_id']=self.custom_brand_id
        CONFIG.mkdir(parents=True,exist_ok=True);tmp=classic.PRESETS.with_suffix('.tmp');tmp.write_text(json.dumps(self.user_presets,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(classic.PRESETS)
        self.group.set('我的预设');self.build_cards();self.status.set('已保存预设：'+name)

    def start(self):
        if self.editor_enabled:return self.editor_start_export()
        if not renderer_available(self.brand.get(),self.look.get()):self.status.set('请先选择滤镜' if self.brand.get()==CUSTOM else GR_UNAVAILABLE);return
        if self.busy or self.custom_importing or not self.files:return
        if not self.denoise_idle():self.status.set('后台降噪正在进行；完成或取消后再导出。');return
        if self.current_post()['denoise'] and self.current_denoise() is None:
            self.status.set('请先点击“开始全尺寸降噪”查看结果，或将降噪强度设为 0 关闭。');return
        try:
            quality=int(self.quality.get());bits=int(self.bits.get());assert 80<=quality<=100 and bits in (8,16) and self.output.get().strip()
        except (ValueError,tk.TclError,AssertionError):messagebox.showerror('导出设置','检查输出文件夹、位深和 JPEG 质量。');return
        self.busy=True;self.run_button.configure(state='disabled');self.video_run.configure(state='disabled');self.progress['value']=0
        self.cancel_export.clear();self.export_cancelled=False
        self.update_brand_controls()
        photo_settings={str(Path(path).resolve()):dict(self.photo_denoise_values.get(str(Path(path).resolve()),
            {'denoise':0.,'denoise_model':'dcu','noise_level':15.})) for path in self.files}
        native_settings={str(Path(path).resolve()):dict(self.photo_native_noise.get(str(Path(path).resolve()),{})) for path in self.files}
        held=self.current_denoise()
        input_records={self.photo_settings_key():held['record']} if held is not None else {}
        args=(list(self.files),Path(self.output.get()),self.signature(),FORMAT_CHOICES[self.format.get()],quality,bits,self.brand.get(),self.current_balance(),self.current_post(),'auto',photo_settings,native_settings,input_records)
        self.export_thread=threading.Thread(target=self.worker,args=args,daemon=True);self.export_thread.start()

    def worker(self,files,folder,sig,fmt,quality,bits,brand=PENTAX,post_balance=(0,0),post_settings=None,backend_mode='auto',photo_settings=None,native_settings=None,input_records=None):
        key,ev,items,amount=sig;errors=[];outputs=[]
        for i,path in enumerate(files):
            try:
                if self.cancel_export.is_set():self.export_cancelled=True;break
                # Stable generated IDs, never arbitrary imported titles, form
                # output filenames. A CUBE key's colon is not a Windows path.
                filename_key=key.replace(':','_') if brand==CUSTOM else ('GR4_' if brand==RICOH else '')+key
                dest=destination_for(path,folder,filename_key,fmt)
                def progress(text):
                    match=re.search(r'(\d+(?:\.\d+)?)%',str(text))
                    fraction=min(.99,float(match.group(1))/100) if match else 0
                    self.events.put(('progress',(100*(i+fraction)/len(files),f'{i+1}/{len(files)} · {text}')))
                from source_export import save as save_source
                if brand==CUSTOM:
                    from custom_lut import save as renderer
                else:renderer=save_gr4 if brand==RICOH else save_creative if key in CREATIVE_LOOKS else save
                settings=validate_post(post_settings)
                identity=str(Path(path).resolve())
                if photo_settings is not None:settings.update(photo_settings[identity])
                controls=dict(items)
                if native_settings is not None:
                    controls={name:value for name,value in controls.items() if name not in NR_CONTROLS}
                    if brand==PENTAX and key in LOOKS:controls.update(native_settings[identity])
                report=save_source(self.photo_denoise_store,renderer,path,dest,key,ev,quality,bits,
                    controls,amount,progress,self.cancel_export.is_set,post_balance,settings,backend_mode,
                    prepared_result=(input_records or {}).get(identity))
                outputs.append({'file':str(dest),'format':report['output_format'],'bits':report['output_bits']})
            except RenderCancelled:self.export_cancelled=True;break
            except Exception as exc:errors.append(Path(path).name+'：'+str(exc))
        self.events.put(('done',(len(outputs),errors,outputs)))

    def report_callback_exception(self,exception,value,tb):
        self.callback_error_count+=1;self.last_callback_error=''.join(traceback.format_exception(exception,value,tb))
        # Runtime-only diagnostics. Never place user paths or these logs in
        # bundled resources, exported photographs or delivery manifests.
        try:
            path=report_dir()/'ui-callback-error.log'
            mode='w' if path.exists() and path.stat().st_size>1024*1024 else 'a'
            with path.open(mode,encoding='utf-8') as stream:
                stream.write('\n'+time.strftime('%Y-%m-%d %H:%M:%S')+'\n'+self.last_callback_error)
        except OSError:pass
        if hasattr(self,'status'):self.status.set('界面更新遇到错误，已记录本机日志；后台状态会继续刷新。')
        observer=self.callback_observer
        if observer:
            try:observer(exception,value,tb)
            except Exception:pass

    def poll(self):
        if self.closed:return
        try:self.poll_once()
        except Exception:self.report_callback_exception(*sys.exc_info())
        finally:
            # A callback failure must not silently stop all future progress and
            # result delivery in a windowed executable without a console.
            if not self.closed:self.root.after(16,self.poll)

    def poll_once(self):
        if self.ready:
            self.poll_custom_import()
            self.poll_denoise()
            # AUDIT-PERF P-1: in editor mode editor_ui.queue_post overrides the
            # legacy path, so post_request stays None and poll_post() can never
            # produce a frame. Skipping it removes an idle call every ~16 ms.
            if not self.editor_enabled:self.poll_post()
            self.viewport.poll()
            if self.denoise_detail is not None:self.denoise_detail.poll()
            self.native_noise_panel.sync()
            self.poll_preset_thumbnails()
            self.poll_editor_tasks()
        try:
            deadline=time.perf_counter()+.008
            while time.perf_counter()<deadline:
                kind,value=self.events.get_nowait()
                if kind=='user_thumb':
                    token,key,fingerprint,thumb=value
                    if token==self.generation and key in self.cards and json.dumps(self.user_presets.get(key[5:],{}),sort_keys=True)==fingerprint:self.set_card_image(key,thumb)
                elif kind=='user_thumb_error':
                    token,key=value
                    if token==self.generation and key in self.cards:self.cards[key][1].configure(text='预览失败 · 可重新选择')
                elif kind=='native_result':
                    token,sig,result=value
                    if token!=self.generation:continue
                    if self.editor_enabled and result[2].get('_editor_input_token')!=self.source_token():continue
                    self.cache_result(token,sig,result);key=sig[0];self.factory[key]=result[2]
                    if self.editor_enabled:self.editor_report=result[2]
                    self.preview_times[key]=round(time.monotonic()-self.photo_started,3)
                    if sig==self.requested:
                        self.show_result(*result)
                        identity=(self.generation,self.brand.get(),self.look.get(),True)
                        if getattr(self,'_controls_identity',None)!=identity:self.build_controls()
                    elif (self.requested and sig[0]==self.requested[0] and self.requested not in self.native_cache
                          and not _is_baseline_signature(sig)):
                        # A valid intermediate image from the same photo/look
                        # is useful while a newer parameter set is calculating.
                        # Keep the pending status; do not claim final settings.
                        self.before=result[0];self.queue_post(result[1]);self.draw()
                    if sig==self.requested and not self.busy and not self.custom_importing and self.denoise_idle():
                        self.status.set('此步骤的降噪结果未缓存，需重新计算' if self.editor_enabled and self.editor_history_missing_denoise() else '预览已更新')
                elif kind=='native_before':
                    token,sig,before=value
                    if token==self.generation and sig==self.requested:self.before=before;self.draw()
                elif kind=='post_stage':
                    serial,token,text=value
                    if token==self.generation and serial==self.post_revision and not self.busy:self.status.set(text)
                elif kind=='denoise_stage':
                    serial,key,text=value
                    if serial==self.denoise_serial and key==self.denoise_key() and self.denoise_future is not None:
                        self.show_denoise_stage(text)
                elif kind=='native_stage':
                    token,sig,text=value
                    if token==self.generation and sig==self.requested:self.preview_stage=text
                elif kind=='native_stats':
                    token,stats=value
                    if token==self.generation:self.preview_stats=stats
                elif kind=='native_error':
                    token,sig,error=value
                    if token!=self.generation:continue
                    if sig is None:self.worker_failed=True
                    if sig is None or sig==self.requested:
                        self.preview_error=error;self.preview_started=0;self.after=None;self.file_info.set('处理失败：'+error);self.status.set(error);self.draw()
                        if self.denoise_waiting:
                            self.fail_denoise_preparation('全尺寸输入准备失败：'+error)
                elif kind=='progress':
                    percent,text=value;self.progress['value']=percent;self.status.set(text)
                elif kind=='done':
                    success,errors,outputs=value;self.busy=False;self.run_button.configure(state='normal');self.video_run.configure(state='normal');self.progress['value']=100;self.last_outputs=outputs
                    self.status.set(f'已导出 {success} 个文件，失败 {len(errors)} 个。'+('其余导出已取消。' if self.export_cancelled else '原片保留。'))
                    self.update_brand_controls()
                    if errors:messagebox.showwarning('部分文件未导出','\n'.join(errors[:12]))
                elif kind=='filters_done':
                    dest,count,error=value;self.busy=False;self.video_run.configure(state='normal');self.update_brand_controls()
                    if error:self.status.set('滤镜文件导出失败');messagebox.showerror('导出失败',error,parent=self.root)
                    else:
                        self.progress['value']=100;self.last_filter_export=dest
                        self.status.set(f'已导出 {count} 个滤镜文件：{dest}')
                        messagebox.showinfo('导出滤镜文件',f'已导出 {count} 个文件到：\n{dest}',parent=self.root)
        except queue.Empty:pass
        if self.preview_started and self.after is None and self.current_path:
            self.file_info.set(f'{Path(self.current_path).name}    处理中 {int(time.monotonic()-self.preview_started)} 秒')

    def help(self):
        messagebox.showinfo('使用说明',
            '照片默认不加滤镜。处理顺序：原图 → 可选 AI 输入降噪 → 照片手动调整 → 所选滤镜。切换滤镜保留当前照片的手动调整。\n\n'
            '照片调色自动保留，切换普通预设、自定义 CUBE 或我的预设都不覆盖它。左侧缩略图保持风格示意，不随调色或降噪刷新。点历史箭头在右侧展开步骤和时间；点击旧步骤只预览，继续编辑时确认是否删除后续步骤。恢复原始还原当前照片。\n\n'
            '降噪方法互斥选择：关闭、DCU 原厂、NAFNet 或 Restormer。AI 使用完整尺寸 RGB16 输入，完成后跨滤镜复用。DCU 四项参数仍在宾得原厂滤镜链内，仅选中宾得原厂滤镜且浓淡 100% 时可用。\n\n'
            '顶部显示后台任务阶段、用时和剩余时间估算，点开可查看排队原因或取消指定任务。降噪结果占比只混合原输入与处理结果，不改变模型强度。\n\n'
            '滚轮围绕鼠标缩放，放大后拖动平移；100% 查看原尺寸，双击回到全图。滑块旁可输入数值，双击复位。\n\n'
            '导出保留完整尺寸；PNG／TIFF 可 16 位无损编码，JPEG 有损。原 RAW 只读，AI 工作副本不是新的相机 RAW。标准 CUBE／XMP 不包含个人降噪或空间调整。\n\n'
            '手动调整是本软件的独立处理；GR4 仍是部分色彩研究模型，尚未验证完整机内一致性。')

    def export_luts(self):
        if self.busy or self.custom_importing:return
        custom=self.brand.get()==CUSTOM
        if custom and (not self.custom_brand_id or not self.custom_library.catalog(brand_id=self.custom_brand_id)):
            self.status.set('当前品牌暂无 CUBE');return
        if self.filter_dialog is not None and self.filter_dialog.winfo_exists():return
        dialog=tk.Toplevel(self.root);self.filter_dialog=dialog
        dialog.title('导出滤镜文件');dialog.configure(bg=BG);dialog.resizable(False,False)
        body=ttk.Frame(dialog,padding=20);body.pack(fill='both',expand=True)
        brand=self.custom_brand_id if custom else self.brand.get()
        brand_name=self.custom_library.brands()[brand] if custom else BRANDS[brand]
        ttk.Label(body,text=brand_name+'滤镜',font=('Microsoft YaHei UI',12,'bold')).pack(anchor='w',pady=(0,10))
        formats=(('cube','CUBE 滤镜（.cube）'),) if custom else (('cube','Photoshop 滤镜（.cube）'),('xmp','Lightroom Classic 配置（.xmp）'))
        self.filter_format_vars={fmt:tk.BooleanVar(master=dialog,value=True) for fmt,label in formats}
        for fmt,label in formats:
            ttk.Checkbutton(body,text=label,variable=self.filter_format_vars[fmt]).pack(anchor='w',pady=4)
        hint=('导出已导入的 CUBE 文件' if custom else '导出全部 34 个标准滤镜：17 个原厂模式、9 个黑白变体、6 个创意组合、2 个基调曲线。\n不含个人预设与临时调整。固定 LUT 无法完整保留原厂照片引擎的所有处理。' if brand==PENTAX else '导出 GR4 的 11 种可用模式、共 17 组色调设置。\n基于当前色彩研究模型采样，不是完整机内引擎；不含个人曝光、浓淡与预设。')
        ttk.Label(body,text=hint,wraplength=430,style='Hint.TLabel').pack(anchor='w',pady=12)
        self.filter_export_submit=ttk.Button(body,text='选择保存位置并导出',style='Accent.TButton',
            command=lambda:self.submit_filter_export(brand))
        self.filter_export_submit.pack(fill='x')

    def submit_filter_export(self,brand):
        if self.closed or self.busy or self.custom_importing:return
        formats=tuple(k for k,v in self.filter_format_vars.items() if v.get())
        if not formats:messagebox.showinfo('选择格式','至少选择一种滤镜文件格式。',parent=self.filter_dialog);return
        try:
            if brand in self.custom_library.brands():
                if formats!=('cube',):raise ValueError('自定义滤镜当前支持 CUBE 导出')
                if not self.custom_library.catalog(brand_id=brand):raise ValueError('当前品牌暂无 CUBE')
            else:require_renderer(brand)
        except ValueError as exc:messagebox.showinfo('暂不可用',str(exc),parent=self.filter_dialog);return
        folder=filedialog.askdirectory(title='选择滤镜文件的保存位置',parent=self.filter_dialog)
        if not folder:return
        self.filter_dialog.destroy();self.filter_dialog=None
        self.busy=True;self.progress['value']=0;self.video_run.configure(state='disabled');self.update_brand_controls()
        def run():
            try:
                dest,files=export_filters(folder,brand,formats,lambda percent,text:self.events.put(('progress',(percent,text))))
                self.events.put(('filters_done',(str(dest),len(files),None)))
            except Exception as exc:self.events.put(('filters_done',(None,0,str(exc))))
        self.export_thread=threading.Thread(target=run,daemon=True);self.export_thread.start()

    def install(self):
        # Preserve the legacy command hook without writing into Adobe folders.
        self.export_luts()

    def export_changed(self):
        super().export_changed()
        self.export_text.set(self.export_text.get().replace('拍摄信息保存在处理记录','不保存拍摄元数据'))

    def start_video(self):
        if self.brand.get()!=PENTAX:self.status.set('此入口支持照片；视频暂未接入。');return
        if not self.output.get().strip():
            messagebox.showerror('导出设置','请选择输出文件夹。');return
        super().start_video()

    def _begin_close(self):
        if getattr(self,'closed',False):return False
        if self.busy:messagebox.showinfo('正在导出','请等待本次导出完成。');return False
        self.closed=True;self.generation+=1
        import startup_prewarm
        startup_prewarm.stop()
        self.preset_thumbnails.cancel()
        self.custom_import_cancel.set()
        for name in ('history_popup','task_popup'):
            window=getattr(self,name,None)
            if window is not None and window.winfo_exists():window.destroy()
            setattr(self,name,None)
        if self.filter_dialog is not None:
            self.filter_dialog.destroy();self.filter_dialog=None
        job=self.denoise_waiting or self.denoise_job
        if job:job[2].set()
        self.denoise_serial+=1;self.denoise_waiting=None
        self.photo_denoise_store.cancel()
        if self.denoise_detail is not None:self.denoise_detail.close()
        self.denoise_progress.stop()
        root=self.root
        for callback in root.tk.call('after','info'):
            # The list includes ttk's pure Tcl progressbar timers. They are
            # not Python command names; after_cancel tries to delete a tuple.
            # Cancel Tcl scheduling now; widget destruction releases registered
            # Python commands on the Tk thread after workers have stopped.
            try:root.tk.call('after','cancel',callback)
            except tk.TclError:pass
        self.viewport.closed=True;self.viewport.serial+=1
        return True

    def _wait_for_close(self):
        # This phase does not call Tk. Workers can finish native cancellation,
        # release file mappings and exit while the main thread keeps pumping.
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.viewport.pool.shutdown(wait=True,cancel_futures=True)
        self.post_pool.shutdown(wait=True,cancel_futures=True)
        self.card_pool.shutdown(wait=True,cancel_futures=True)
        self.preset_thumbnails.close(wait=True)
        self.denoise_pool.shutdown(wait=True,cancel_futures=True)
        self.custom_import_pool.shutdown(wait=True,cancel_futures=True)
        self.photo_denoise_store.close(wait=True)
        if self.denoise_detail is not None:
            self.denoise_detail.pool.shutdown(wait=True,cancel_futures=True)
            self.denoise_detail.display_pool.shutdown(wait=True,cancel_futures=True)
        neural=sys.modules.get('neural_process')
        import startup_prewarm
        startup_prewarm.stop(wait=True)
        if neural is not None:neural.shutdown()
        if self.export_thread and self.export_thread.is_alive():self.export_thread.join()

    def _finish_close(self):
        # Callbacks scheduled during orderly shutdown must not outlive Tcl.
        for callback in self.root.tk.call('after','info'):
            try:self.root.tk.call('after','cancel',callback)
            except tk.TclError:pass
        self.root.destroy()
        # Widgets, Tk variables and image callbacks form cycles. Release them
        # on the GUI thread after every worker exits, including close/reopen.
        self.__dict__.clear();self.closed=True
        import gc
        gc.collect()

    def request_close(self):
        """Interactive close: hide promptly and keep Tk alive during cleanup."""
        if not self._begin_close():return
        self.root.withdraw()
        cleanup=threading.Thread(target=self._wait_for_close,name='PhotoShutdown',daemon=False)
        cleanup.start()
        def finish_when_ready():
            if cleanup.is_alive():self.root.after(20,finish_when_ready)
            else:
                cleanup.join();self._finish_close()
        self.root.after(20,finish_when_ready)

    def close(self):
        # Synchronous teardown remains available to noninteractive test callers.
        if self._begin_close():self._wait_for_close();self._finish_close()


_SINGLE_INSTANCE=None

def _claim_single_instance():
    """Return False when another GUI instance already owns the mutex."""
    if os.name != 'nt':return True
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.CreateMutexW.restype=wintypes.HANDLE
    handle=kernel.CreateMutexW(None,False,'Local\\PhotoWorkbenchAppInstance')
    if not handle:return True
    if ctypes.get_last_error()==183:
        from ui_language import messagebox
        messagebox.showinfo('已在运行中',
            '程序已经在运行中。\n\n请切换到已打开的窗口，或先关闭它再重新启动。')
        return False
    global _SINGLE_INSTANCE
    _SINGLE_INSTANCE=handle
    return True


def main():
    if '--native-worker' in sys.argv:
        from native_worker import main as worker
        raise SystemExit(worker(sys.argv[sys.argv.index('--native-worker')+1]))
    if '--verify-native' in sys.argv:
        from verify_native import main as verify
        verify(Path(sys.argv[sys.argv.index('--verify-native')+1]));return
    if '--verify-upgrade' in sys.argv:
        from verify_upgrade import main as verify
        verify(Path(sys.argv[sys.argv.index('--verify-upgrade')+1]));return
    if '--verify-workbench' in sys.argv:
        from verify_workbench import main as verify
        verify(Path(sys.argv[sys.argv.index('--verify-workbench')+1]));return
    if '--verify-delivery' in sys.argv:
        from verify_delivery import main as verify
        verify(Path(sys.argv[sys.argv.index('--verify-delivery')+1]));return
    if '--verify-brand-export' in sys.argv:
        from verify_brand_export import main as verify
        verify(Path(sys.argv[sys.argv.index('--verify-brand-export')+1]));return
    if '--verify-gr4' in sys.argv:
        from verify_gr4 import main as verify
        verify(Path(sys.argv[sys.argv.index('--verify-gr4')+1]));return
    if '--verify-gr4-files' in sys.argv:
        from verify_gr4_files import main as verify
        index=sys.argv.index('--verify-gr4-files')
        arguments=sys.argv[index+2:]
        if '--report-dir' in arguments:arguments=arguments[:arguments.index('--report-dir')]
        verify(Path(sys.argv[index+1]),arguments);return
    if '--render-photo' in sys.argv:
        i=sys.argv.index('--render-photo');save(sys.argv[i+1],sys.argv[i+2],sys.argv[i+3] if len(sys.argv)>i+3 else 'Natural');return
    # AUDIT-P25: the launcher guards its own launches with a mutex, but a
    # second copy of the unpacked payload would still start if the launcher
    # died while this process kept running. Guard the GUI entry too.
    if not _claim_single_instance():return
    from startup_signal import after_first_frame
    root=tk.Tk();app=App(root)
    root.after(40,lambda:after_first_frame(root,app.start_background_preparation))
    root.mainloop()


if __name__=='__main__':
    try:main()
    except Exception:
        (report_dir()/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        if not any(x.startswith('--') for x in sys.argv[1:]):messagebox.showerror('启动失败','启动失败，详情已写入本机日志。')
        sys.exit(1)
