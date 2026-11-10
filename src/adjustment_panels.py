"""Fine, collapsible manual adjustments; the original renderer stays separate."""
import tkinter as tk
from tkinter import ttk
from basic_adjustments import validate as validate_basic
from color_mixer import COLORS, NAMES, SWATCHES, validate as validate_mixer
from ui_controls import FineSlider


class BasicPanel(ttk.Frame):
    def __init__(self, parent, command):
        super().__init__(parent)
        self.command=command;self.syncing=False;self.enabled=True;self.values={};self.sliders={}
        self.regions=[tk.DoubleVar(self,value=0) for _ in range(5)]
        self.region_sliders=[]
        for title,rows in (
            ('白平衡',(('warmth','冷暖',('#648fff','#f2c575')),('tint','色调',('#77bd92','#ca7caf')))),
            ('明暗',(('contrast','对比度',None),)),
        ):
            self.heading(title)
            for key,label,gradient in rows:self.add_slider(self,key,label,gradient)
        for index,label in ((3,'高光'),(1,'阴影'),(4,'白色'),(0,'黑色'),(2,'中间调')):
            ttk.Label(self,text=label,style='Hint.TLabel').pack(anchor='w',pady=(3,0))
            slider=FineSlider(self,variable=self.regions[index],from_=-100,to=100,
                              resolution=1,default=0,command=lambda value:self.changed())
            slider.pack(fill='x');self.region_sliders.append(slider)
        self.heading('质感与色彩')
        for key,label in (('texture','纹理'),('clarity','清晰度'),('dehaze','去朦胧'),
                          ('vibrance','自然饱和度'),('saturation','饱和度')):
            self.add_slider(self,key,label)
        ttk.Button(self,text='复位基础调整',command=self.reset).pack(fill='x',pady=(7,3))

    def heading(self,text):
        ttk.Label(self,text=text,font=('Microsoft YaHei UI',9,'bold')).pack(anchor='w',pady=(8,4))

    def add_slider(self,parent,key,label,gradient=None,lo=-100,hi=100):
        ttk.Label(parent,text=label,style='Hint.TLabel').pack(anchor='w',pady=(3,0))
        variable=tk.DoubleVar(self,value=0);self.values[key]=variable
        slider=FineSlider(parent,variable=variable,from_=lo,to=hi,resolution=1,default=0,
                          gradient=gradient,command=lambda value:self.changed())
        slider.pack(fill='x');self.sliders[key]=slider

    def build_detail(self,parent):
        self.add_slider(parent,'sharpen','锐化',lo=0)
        self.add_slider(parent,'soften','全图柔化',lo=0)
        ttk.Button(parent,text='复位细节调整',command=self.reset_detail).pack(fill='x',pady=4)

    def set(self,basic,regions):
        basic=validate_basic(basic);self.syncing=True
        try:
            for key,variable in self.values.items():variable.set(basic[key])
            for variable,value in zip(self.regions,regions):variable.set(round(value*50,8))
        finally:self.syncing=False

    def changed(self):
        if not self.syncing and self.enabled:
            self.command({key:variable.get() for key,variable in self.values.items()},
                         tuple(variable.get()/50 for variable in self.regions))

    def reset(self):
        if not self.enabled:return
        for key,variable in self.values.items():
            if key not in ('sharpen','soften'):variable.set(0)
        for variable in self.regions:variable.set(0)
        self.changed()

    def reset_detail(self):
        if not self.enabled:return
        for key in ('sharpen','soften'):self.values[key].set(0)
        self.changed()

    def set_enabled(self,enabled):
        self.enabled=enabled
        for slider in (*self.sliders.values(),*self.region_sliders):slider.configure(state='normal' if enabled else 'disabled')


class ColorMixerPanel(ttk.Frame):
    def __init__(self,parent,command):
        super().__init__(parent);self.command=command;self.value=validate_mixer()
        self.channel='hue';self.values={};self.sliders={};self.buttons={};self.enabled=True
        bar=ttk.Frame(self);bar.pack(fill='x',pady=(4,7))
        for index,(key,label) in enumerate((('hue','色相'),('saturation','饱和度'),('luminance','明亮度'))):
            bar.columnconfigure(index,weight=1)
            button=ttk.Button(bar,text=label,width=0,command=lambda k=key:self.select(k))
            button.grid(row=0,column=index,sticky='ew',padx=(0,3));self.buttons[key]=button
        for key in COLORS:
            row=ttk.Frame(self);row.pack(fill='x',pady=(3,0))
            dot=tk.Canvas(row,width=16,height=16,bg='#1c1c1e',highlightthickness=0)
            dot.pack(side='left');dot.create_oval(4,4,12,12,outline='',fill=SWATCHES[COLORS.index(key)])
            ttk.Label(row,text=NAMES[COLORS.index(key)],style='Hint.TLabel').pack(side='left',padx=(3,0))
            variable=tk.DoubleVar(self,value=0);self.values[key]=variable
            slider=FineSlider(self,variable=variable,from_=-100,to=100,default=0,resolution=1,
                              command=lambda value,k=key:self.changed(k))
            slider.pack(fill='x');self.sliders[key]=slider
        ttk.Button(self,text='复位混色',command=self.reset).pack(fill='x',pady=(7,4))
        self.select('hue')

    def select(self,channel):
        self.channel=channel
        for key,button in self.buttons.items():button.configure(style='Accent.TButton' if key==channel else 'TButton')
        for key,variable in self.values.items():variable.set(self.value[key][channel])

    def changed(self,key):
        if not self.enabled:return
        self.value[key][self.channel]=self.values[key].get()
        self.command(validate_mixer(self.value))

    def set(self,value):self.value=validate_mixer(value);self.select(self.channel)
    def reset(self):
        if self.enabled:self.set(None);self.command(validate_mixer(self.value))

    def set_enabled(self,enabled):
        self.enabled=enabled
        for slider in self.sliders.values():slider.configure(state='normal' if enabled else 'disabled')
