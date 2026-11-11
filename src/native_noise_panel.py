"""Expose only the four proven controls of DCU's original development path."""
from ui_language import tk, ttk
from native_controls import NR_CONTROLS
from native_backend import LOOKS
from filter_catalog import PENTAX
from ui_controls import FineSlider, Tooltip


class NativeNoisePanel(ttk.Frame):
    def __init__(self,app,parent):
        super().__init__(parent);self.app=app;self.pack(fill='x',pady=(0,8))
        self.enabled=tk.BooleanVar(master=self,value=False);self.values={};self.labels={};self.syncing=False
        self.key=None;self.sliders={}
        self.toggle=ttk.Checkbutton(self,text='自定义 DCU 原厂降噪',variable=self.enabled,command=self.changed)
        self.toggle.pack(anchor='w')
        self.note=tk.StringVar(master=self,value='默认设置')
        ttk.Label(self,textvariable=self.note,wraplength=250,style='Hint.TLabel').pack(fill='x')
        self.body=ttk.Frame(self)
        for key,meta in NR_CONTROLS.items():
            row=ttk.Frame(self.body);row.pack(fill='x',pady=(5,0))
            ttk.Label(row,text=meta['name_zh']).pack(side='left')
            label=tk.StringVar(master=self,value=str(meta['default']));self.labels[key]=label
            ttk.Label(row,textvariable=label).pack(side='right')
            value=tk.DoubleVar(master=self,value=meta['default']);self.values[key]=value
            slider=FineSlider(self.body,variable=value,from_=meta['min'],to=meta['max'],
                              resolution=1,default=meta['default'],command=lambda v,k=key:self.changed(k,v))
            slider.pack(fill='x');self.sliders[key]=slider
            Tooltip(slider,meta['tooltip'])
        self.reset_button=ttk.Button(self.body,text='恢复原有显影设置',command=self.reset)
        self.reset_button.pack(fill='x',pady=(5,0))

    def available(self):
        a=self.app
        # Wait for the current input's actual factory values. RGB adaptation
        # can use defaults different from the class constructor's 0/0/80/0.
        ready=bool(a.factory.get(a.look.get(),{}).get('factory_noise_reduction'))
        return a.brand.get()==PENTAX and a.look.get() in LOOKS and ready and not a.busy

    def reset(self):self.enabled.set(False);self.changed()

    def changed(self,field=None,value=None):
        if self.syncing or not self.app.ready or not self.available():return
        a=self.app;look=a.look.get();controls=dict(a.native_noise_values())
        if field is not None:
            number=round(float(value));self.values[field].set(number);self.labels[field].set(str(number))
        for key,meta in NR_CONTROLS.items():
            if self.enabled.get():controls[key]=max(meta['min'],min(meta['max'],round(self.values[key].get())))
            else:controls.pop(key,None)
        a.photo_native_noise[a.photo_settings_key()]=controls;self.key=None;self.sync()
        if a.denoise_detail is not None:a.denoise_detail.set_source_mode('native')
        a.settings_changed()

    def sync(self):
        a=self.app
        if not a.ready:return
        valid=self.available();controls=a.native_noise_values() if valid else {}
        custom=any(key in controls for key in NR_CONTROLS)
        factory=a.factory.get(a.look.get(),{}).get('factory_noise_reduction',{}) if valid else {}
        numbers={key:controls.get(key,factory.get(key,{}).get('value',meta['default'])) for key,meta in NR_CONTROLS.items()}
        signature=(a.brand.get(),a.look.get(),valid,custom,tuple(numbers.items()))
        if signature==self.key:return
        self.key=signature;self.syncing=True
        try:
            self.toggle.configure(state='normal' if valid else 'disabled');self.enabled.set(custom)
            for key,value in numbers.items():
                self.values[key].set(value);self.labels[key].set(str(value))
                self.sliders[key].configure(default=factory.get(key,{}).get('value',NR_CONTROLS[key]['default']),state='normal' if valid else 'disabled')
            if valid and custom:
                self.body.pack(fill='x');self.note.set('自定义')
            else:
                self.body.pack_forget()
                self.note.set('默认设置' if valid else
                    '导出中' if a.busy else
                    '等待预览' if a.brand.get()==PENTAX and a.look.get() in LOOKS else
                    '当前滤镜不可用')
        finally:self.syncing=False
