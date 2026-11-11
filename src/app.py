"""Offline photo editor with real-image preset previews and explicit exports."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, os, queue, shutil, sys, threading, traceback
from ui_language import tk, ttk, filedialog, messagebox, simpledialog
import numpy as np
from PIL import Image, ImageTk
from ui_controls import FineSlider
from engine import ROOT, LOOKS, apply, srgb_encode
from process_photo import (process, read_photo, adjust_rgb, destination_for, resolve_format,
                           FORMAT_CHOICES, PHOTO_EXT, RAW_EXT)
from process_video import render
from app_paths import (VERSION, CONFIG, DEFAULT_OUTPUT, report_dir, resource_names,
                       resource_bytes, resource_file, export_resources, ffmpeg_path)

PRESETS=CONFIG/'presets.json'
APP_NAME='RICOH & PENTAX SOOC Simulator'
DEFAULT_ORDER=['GOLD','SATOBI','Bright','Portrait','Landscape','MIYABI','Muted','Reversal','BleachBypass','Radiant','Flat']
DESCRIPTIONS={
 'GOLD':'金色风格 · 矩阵与通道曲线组合，点击图片比较实际变化。',
 'SATOBI':'里美风格 · 观察当前照片的肤色、蓝色与绿色变化。',
 'Bright':'鲜明 · 对比当前照片的色彩浓度。',
 'Portrait':'人像 · 重点观察肤色和明暗过渡。',
 'Landscape':'风景 · 重点观察天空、绿植和远景。',
 'MIYABI':'风雅 · 重点观察红绿等色彩的变化。',
 'Muted':'原名温暖 / Muted：降低整体饱和度、保留主色倾向，并非调暖色温。当前移植的减淡程度尚未与 DCU 完整输出验证。',
 'Reversal':'反转片 · 包含反转片基调映射。',
 'BleachBypass':'留银冲洗 · 观察低饱和色彩变化。',
 'Radiant':'波普 · 观察鲜艳色彩的变化。',
 'Flat':'平面 · 对比颜色变化，不承诺保留 RAW 高光。',
}
def preset_description(key):
    if key in DESCRIPTIONS:return DESCRIPTIONS[key]
    if key.startswith('Mono'):return '黑白通道混合 · 各版本对不同颜色的明暗转换不同，直接对比缩略图选择。'
    if key.startswith('CP_'):return '交叉处理 · 本工具的曲线通道组合，非已验证的原厂预设。'
    return '单独的基调曲线效果。'
def label_for(key):
    if key=='Muted':return '温暖 · 低饱和（Muted）'
    name=LOOKS[key]['name']
    return key if key==name else name
def install_profiles():
    target=Path(os.environ['APPDATA'])/'Adobe/CameraRaw/Settings/DCU5_DataTransfer_v2'
    target.mkdir(parents=True,exist_ok=True);count=0
    for name in resource_names('exports/Lightroom_Profiles'):
        if not name.endswith('.xmp'):continue
        data=resource_bytes(name);dest=target/Path(name).name
        if dest.exists():
            if dest.read_bytes()!=data:raise ValueError('同名配置文件已有修改，未覆盖：'+str(dest))
        else:
            with dest.open('xb') as handle:handle.write(data)
        count+=1
    return target,count
def preview_input(path,maxsize=(1200,900)):
    photo=read_photo(path,half_size=Path(path).suffix.lower() in RAW_EXT)
    rgb=photo.rgb.astype(np.float32)/np.iinfo(photo.rgb.dtype).max
    h,w=rgb.shape[:2];ratio=min(1,maxsize[0]/w,maxsize[1]/h)
    size=(max(1,round(w*ratio)),max(1,round(h*ratio)))
    if ratio<1:
        rgb=np.stack([np.array(Image.fromarray(rgb[...,i]).resize(size,Image.Resampling.BILINEAR)) for i in range(3)],axis=-1)
    alpha=None
    if photo.alpha is not None:
        alpha=photo.alpha.astype(np.float32)/np.iinfo(photo.alpha.dtype).max
        if ratio<1:alpha=np.array(Image.fromarray(alpha).resize(size,Image.Resampling.BILINEAR))
    return rgb,photo.linear,alpha,(photo.rgb.shape[1],photo.rgb.shape[0]),photo.rgb.dtype.itemsize*8
def to_image(rgb,alpha=None):
    if alpha is not None:
        y,x=np.indices(rgb.shape[:2]);checker=np.where((x//12+y//12)%2,.84,.96)[...,None]
        rgb=rgb*alpha[...,None]+checker*(1-alpha[...,None])
    return Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8))

class App:
    def __init__(self,root):
        self.root=root;self.files=[];self.events=queue.Queue();self.busy=False
        self.pool=ThreadPoolExecutor(max_workers=2);self.generation=0;self.render_id=0
        self.preview_data=None;self.before=None;self.after=None;self.split=.5;self.pending=None
        self.cards={};self.thumbnails={};self._photos=[];self.current_path=None
        self.user_presets={}
        try:self.user_presets=json.loads(PRESETS.read_text(encoding='utf-8'))
        except (OSError,ValueError):pass
        if not isinstance(self.user_presets,dict):self.user_presets={}
        self.user_presets={k:v for k,v in self.user_presets.items() if isinstance(v,dict) and v.get('look') in LOOKS}
        root.title(APP_NAME+' '+VERSION)
        root.minsize(1100,760);root.configure(bg='#f4f5f3')
        # AUDIT-P29: open maximized. 'zoomed' keeps the title bar and the restore
        # button, unlike -fullscreen which hides the frame and can trap the user.
        root.geometry('1280x860')
        root.state('zoomed')
        style=ttk.Style(root);style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10),background='#f4f5f3')
        style.configure('TButton',padding=(9,6))
        style.configure('Accent.TButton',background='#17644e',foreground='white',font=('Microsoft YaHei UI',10,'bold'))
        style.map('Accent.TButton',background=[('active','#227e64'),('disabled','#869b94')])
        style.configure('Title.TLabel',font=('Microsoft YaHei UI',18,'bold'))
        style.configure('Hint.TLabel',foreground='#5e6863',font=('Microsoft YaHei UI',9))
        header=ttk.Frame(root,padding=(18,12));header.pack(fill='x');self.header=header
        brand=ttk.Frame(header);brand.pack(side='left',padx=(0,14))
        ttk.Label(brand,text='RICOH & PENTAX',style='Title.TLabel').pack(anchor='w')
        ttk.Label(brand,text='SOOC Simulator  '+VERSION,style='Hint.TLabel').pack(anchor='w')
        ttk.Button(header,text='使用说明',command=self.help).pack(side='right')
        ttk.Button(header,text='安装 Lightroom 配置',command=self.install).pack(side='right',padx=8)
        ttk.Button(header,text='导出 LUT / 配置',command=self.export_luts).pack(side='right')
        self.tabs=ttk.Notebook(root);self.tabs.pack(fill='both',expand=True,padx=14)
        photo=ttk.Frame(self.tabs,padding=10);video=ttk.Frame(self.tabs,padding=18)
        self.tabs.add(photo,text='  照片  ');self.tabs.add(video,text='  视频  ')
        self.look=tk.StringVar(value='GOLD');self.amount=tk.DoubleVar(value=100);self.exposure=tk.DoubleVar(value=0)
        self.format=tk.StringVar(value='跟随输入格式');self.quality=tk.IntVar(value=98);self.bits=tk.StringVar(value='16')
        self.output=tk.StringVar(value=str(DEFAULT_OUTPUT));self.view=tk.StringVar(value='split')
        self.group=tk.StringVar(value='彩色');self.status=tk.StringVar(value='就绪')
        self.preset_text=tk.StringVar();self.export_text=tk.StringVar();self.file_info=tk.StringVar(value='未选择照片')
        self.amount_text=tk.StringVar();self.ev_text=tk.StringVar()
        tools=ttk.Frame(photo);tools.pack(fill='x',pady=(0,10))
        ttk.Button(tools,text='＋ 添加照片 / RAW',command=self.choose,style='Accent.TButton').pack(side='left')
        ttk.Button(tools,text='添加文件夹',command=self.choose_folder).pack(side='left',padx=8)
        ttk.Button(tools,text='清空列表',command=self.clear).pack(side='left')
        ttk.Label(tools,text='JPEG / PNG / TIFF / WebP / BMP / RAW',style='Hint.TLabel').pack(side='right')
        body=ttk.Frame(photo);body.pack(fill='both',expand=True)
        body.columnconfigure(1,weight=1);body.rowconfigure(0,weight=1)
        left=ttk.Frame(body,width=274);left.grid(row=0,column=0,sticky='nsew',padx=(0,10));left.grid_propagate(False)
        left.columnconfigure(0,weight=1);left.rowconfigure(4,weight=1)
        ttk.Label(left,text='照片').grid(row=0,column=0,sticky='w')
        self.file_list=tk.Listbox(left,height=5,exportselection=False,font=('Microsoft YaHei UI',9),borderwidth=0,selectbackground='#17644e')
        self.file_list.grid(row=1,column=0,sticky='ew',pady=(6,12))
        self.file_list.bind('<<ListboxSelect>>',self.file_selected)
        ttk.Label(left,text='预设 · 缩略图为标准效果').grid(row=2,column=0,sticky='w')
        groups=ttk.Combobox(left,textvariable=self.group,values=['彩色','黑白','创意组合','基调曲线','我的预设'],state='readonly')
        self.group_combo=groups
        groups.grid(row=3,column=0,sticky='ew',pady=6);groups.bind('<<ComboboxSelected>>',lambda e:self.build_cards())
        gridbox=ttk.Frame(left);gridbox.grid(row=4,column=0,sticky='nsew')
        self.preset_canvas=tk.Canvas(gridbox,width=250,bg='#f4f5f3',highlightthickness=0)
        self.preset_canvas.pack(side='left',fill='both',expand=True)
        sc=ttk.Scrollbar(gridbox,orient='vertical',command=self.preset_canvas.yview);sc.pack(side='right',fill='y')
        self.preset_canvas.configure(yscrollcommand=sc.set)
        self.card_frame=ttk.Frame(self.preset_canvas);self.preset_canvas.create_window((0,0),window=self.card_frame,anchor='nw')
        self.card_frame.bind('<Configure>',lambda e:self.preset_canvas.configure(scrollregion=self.preset_canvas.bbox('all')))
        center=ttk.Frame(body);center.grid(row=0,column=1,sticky='nsew',padx=(0,12))
        views=ttk.Frame(center);views.pack(fill='x')
        for text,value in [('左右对比','split'),('处理后','after'),('原图','before')]:
            ttk.Radiobutton(views,text=text,variable=self.view,value=value,command=self.draw).pack(side='left',padx=(0,10))
        ttk.Label(center,textvariable=self.file_info,style='Hint.TLabel',wraplength=480).pack(anchor='w',pady=8)
        self.canvas=tk.Canvas(center,bg='#242a28',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.draw())
        self.canvas.bind('<B1-Motion>',self.drag);self.canvas.bind('<Button-1>',self.drag)
        right_host=ttk.Frame(body,width=300);right_host.grid(row=0,column=2,sticky='nsew')
        right_canvas=tk.Canvas(right_host,width=282,bg='#f4f5f3',highlightthickness=0)
        right_canvas.pack(side='left',fill='both',expand=True)
        right_scroll=ttk.Scrollbar(right_host,orient='vertical',command=right_canvas.yview)
        right_scroll.pack(side='right',fill='y');right_canvas.configure(yscrollcommand=right_scroll.set)
        right=ttk.Frame(right_canvas,width=282)
        right_window=right_canvas.create_window((0,0),window=right,anchor='nw',width=282)
        right.bind('<Configure>',lambda e:right_canvas.configure(scrollregion=right_canvas.bbox('all')))
        right_canvas.bind('<Configure>',lambda e:right_canvas.itemconfigure(right_window,width=e.width))
        ttk.Label(right,text='调色',font=('Microsoft YaHei UI',12,'bold')).pack(anchor='w')
        ttk.Label(right,textvariable=self.preset_text,wraplength=278,style='Hint.TLabel').pack(anchor='w',pady=(6,12))
        ttk.Label(right,textvariable=self.amount_text).pack(anchor='w')
        FineSlider(right,variable=self.amount,from_=0,to=200,resolution=1,default=100,command=lambda v:self.settings_changed()).pack(fill='x',pady=3)
        amounts=ttk.Frame(right);amounts.pack(fill='x')
        for text,value in [('不加',0),('轻柔',50),('标准',100),('加强',150)]:
            ttk.Button(amounts,text=text,width=0,command=lambda v=value:self.set_amount(v)).pack(side='left',padx=(0,3))
        ttk.Label(right,textvariable=self.ev_text).pack(anchor='w')
        FineSlider(right,variable=self.exposure,from_=-3,to=3,resolution=.01,default=0,command=lambda v:self.settings_changed()).pack(fill='x')
        pr=ttk.Frame(right);pr.pack(fill='x',pady=(4,12))
        ttk.Button(pr,text='重置调色',command=self.reset).pack(side='left')
        ttk.Button(pr,text='保存我的预设',command=self.save_preset).pack(side='left',padx=6)
        ttk.Separator(right).pack(fill='x',pady=(0,10))
        ttk.Label(right,text='导出',font=('Microsoft YaHei UI',12,'bold')).pack(anchor='w')
        self.format_combo=ttk.Combobox(right,textvariable=self.format,values=list(FORMAT_CHOICES),state='readonly')
        self.format_combo.pack(fill='x',pady=8);self.format_combo.bind('<<ComboboxSelected>>',lambda e:self.export_changed())
        row=ttk.Frame(right);row.pack(fill='x')
        ttk.Label(row,text='JPEG 质量').pack(side='left')
        self.quality_spin=ttk.Spinbox(row,from_=80,to=100,textvariable=self.quality,width=5,command=self.export_changed)
        self.quality_spin.pack(side='left',padx=6)
        ttk.Label(row,text='位深').pack(side='left')
        self.bits_combo=ttk.Combobox(row,textvariable=self.bits,values=['16','8'],width=4,state='readonly')
        self.bits_combo.pack(side='left',padx=6);self.bits_combo.bind('<<ComboboxSelected>>',lambda e:self.export_changed())
        ttk.Label(right,textvariable=self.export_text,wraplength=278,style='Hint.TLabel').pack(anchor='w',pady=8)
        ttk.Entry(right,textvariable=self.output).pack(fill='x')
        row=ttk.Frame(right);row.pack(fill='x',pady=6)
        ttk.Button(row,text='选择输出文件夹',command=self.choose_output).pack(side='left')
        ttk.Button(row,text='打开文件夹',command=self.open_output).pack(side='left',padx=4)
        self.run_button=ttk.Button(right,text='导出列表中的全部照片',style='Accent.TButton',command=self.start)
        self.run_button.pack(fill='x',pady=8)
        self.progress=ttk.Progressbar(right,maximum=100);self.progress.pack(fill='x')
        self.make_video_tab(video)
        footer=ttk.Frame(root,padding=(18,8));footer.pack(fill='x')
        ttk.Label(footer,textvariable=self.status,wraplength=1200).pack(anchor='w')
        self.build_cards();self.settings_changed();self.export_changed()
        root.after(80,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def build_cards(self):
        for widget in self.card_frame.winfo_children():widget.destroy()
        self.cards={}
        if self.group.get()=='彩色':keys=DEFAULT_ORDER
        elif self.group.get()=='黑白':keys=[k for k in LOOKS if k.startswith('Mono')]
        elif self.group.get()=='创意组合':keys=[k for k in LOOKS if k.startswith('CP_')]
        elif self.group.get()=='基调曲线':keys=['BaseTone','ReversalTone']
        else:keys=['user:'+k for k in self.user_presets]
        if not keys:
            ttk.Label(self.card_frame,text='调好浓淡和曝光后，点击\n“保存我的预设”。').pack(pady=16);return
        for i,key in enumerate(keys):
            user=key.startswith('user:');data=self.user_presets[key[5:]] if user else None
            look=data['look'] if user else key
            title=key[5:] if user else label_for(key)
            frame=tk.Frame(self.card_frame,bg='#ffffff',highlightthickness=2,highlightbackground='#dce2de')
            frame.grid(row=i//2,column=i%2,padx=3,pady=4,sticky='n')
            label=tk.Label(frame,text='添加照片\n查看效果',width=15,height=5,bg='#e8ece8',fg='#5e6863')
            label.pack()
            tk.Label(frame,text=title,font=('Microsoft YaHei UI',9),wraplength=112,width=15,bg='white').pack(pady=4)
            for widget in [frame,*frame.winfo_children()]:widget.bind('<Button-1>',lambda e,k=key:self.select_preset(k))
            self.cards[key]=(frame,label)
            if not user and look in self.thumbnails:self.set_card_image(key,self.thumbnails[look])
            elif user and self.preview_data is not None:
                rgb,linear,alpha,*_=self.preview_data
                im=to_image(adjust_rgb(rgb,linear,look,data['amount']/100,data['exposure']),alpha)
                im.thumbnail((112,76),Image.Resampling.LANCZOS);self.set_card_image(key,im)
        self.highlight()
    def set_card_image(self,key,im):
        if key not in self.cards:return
        photo=ImageTk.PhotoImage(im);label=self.cards[key][1]
        label.configure(image=photo,text='',width=112,height=76);label.image=photo
    def highlight(self):
        for key,(frame,_) in self.cards.items():frame.configure(highlightbackground='#17644e' if key==self.look.get() else '#dce2de')
    def select_preset(self,key):
        if key.startswith('user:'):
            data=self.user_presets[key[5:]];self.look.set(data['look']);self.amount.set(data['amount']);self.exposure.set(data['exposure'])
        else:self.look.set(key)
        self.highlight();self.settings_changed()
    def set_amount(self,value):self.amount.set(value);self.settings_changed()
    def reset(self):self.amount.set(100);self.exposure.set(0);self.settings_changed()
    def save_preset(self):
        name=simpledialog.askstring('保存我的预设','输入名称；保存当前滤镜、浓淡和曝光。',parent=self.root)
        if not name:return
        name=name.strip()[:40]
        if not name:return
        if name in self.user_presets and not messagebox.askyesno('同名预设','替换已保存的同名预设？'):return
        self.user_presets[name]={'look':self.look.get(),'amount':round(self.amount.get(),1),'exposure':round(self.exposure.get(),2)}
        CONFIG.mkdir(parents=True,exist_ok=True)
        temp=PRESETS.with_suffix('.tmp');temp.write_text(json.dumps(self.user_presets,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(PRESETS)
        self.group.set('我的预设');self.build_cards();self.status.set('已保存预设：'+name)
    def choose(self):
        if self.busy:return
        paths=filedialog.askopenfilenames(title='添加照片',filetypes=[('照片 / RAW',' '.join('*'+e for e in sorted(PHOTO_EXT))),('所有文件','*.*')])
        self.add_files(paths)
    def choose_folder(self):
        if self.busy:return
        folder=filedialog.askdirectory(title='添加文件夹中的照片（不含子文件夹）')
        if folder:self.add_files(sorted(str(p) for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in PHOTO_EXT))
    def add_files(self,paths):
        rejected=[p for p in paths if Path(p).suffix.lower() not in PHOTO_EXT]
        if rejected:messagebox.showwarning('不支持此格式','目前支持 JPEG、PNG、TIFF、WebP、BMP 和相机 RAW。\n未添加：\n'+'\n'.join(Path(p).name for p in rejected))
        for path in paths:
            if Path(path).suffix.lower() not in PHOTO_EXT:continue
            path=str(Path(path).resolve())
            if path not in self.files:self.files.append(path);self.file_list.insert('end',Path(path).name)
        if self.files and not self.file_list.curselection():
            self.file_list.selection_set(0);self.file_selected()
        self.run_button.configure(text=f'导出全部 {len(self.files)} 张照片')
        self.export_changed()
    def clear(self):
        if self.busy:return
        self.files=[];self.file_list.delete(0,'end');self.generation+=1;self.render_id+=1
        self.preview_data=None;self.before=None;self.after=None;self.current_path=None;self.thumbnails={}
        self.file_info.set('未选择照片');self.build_cards();self.draw();self.export_changed()
        self.run_button.configure(text='导出列表中的全部照片')
    def file_selected(self,event=None):
        indices=self.file_list.curselection()
        if not indices:return
        path=self.files[indices[0]]
        if self.current_path==path:return
        self.current_path=path;self.generation+=1;token=self.generation;self.render_id+=1
        self.preview_data=None;self.before=None;self.after=None;self.thumbnails={}
        self.build_cards();self.draw();self.file_info.set(Path(path).name+' · 正在生成预览…')
        self.pool.submit(self.load_worker,path,token);self.export_changed()
    def load_worker(self,path,token):
        try:
            data=preview_input(path);rgb,linear,alpha,*_=data
            self.events.put(('loaded',(token,data)))
            h,w=rgb.shape[:2];step=max(1,min(h//76,w//112));small=rgb[::step,::step];a=alpha[::step,::step] if alpha is not None else None
            thumbs={}
            for key in LOOKS:
                if token!=self.generation:return
                im=to_image(adjust_rgb(small,linear,key),a);im.thumbnail((112,76),Image.Resampling.LANCZOS);thumbs[key]=im
            self.events.put(('thumbs',(token,thumbs)))
        except Exception as e:self.events.put(('load_error',(token,str(e))))
    def settings_changed(self):
        self.amount_text.set(f'滤镜浓淡   {self.amount.get():.0f}%')
        self.ev_text.set(f'曝光补偿   {self.exposure.get():+.2f} EV')
        self.preset_text.set(label_for(self.look.get())+'\n'+preset_description(self.look.get()))
        if self.pending:self.root.after_cancel(self.pending)
        self.pending=self.root.after(180,self.request_render)
    def request_render(self):
        self.pending=None
        if self.preview_data is None:return
        self.render_id+=1;rid=self.render_id;token=self.generation
        self.pool.submit(self.render_worker,token,rid,self.preview_data,self.look.get(),self.amount.get()/100,self.exposure.get())
    def render_worker(self,token,rid,data,look,amount,ev):
        try:
            rgb,linear,alpha,*_=data
            after=to_image(adjust_rgb(rgb,linear,look,amount,ev),alpha)
            self.events.put(('rendered',(token,rid,after)))
        except Exception as e:self.events.put(('load_error',(token,str(e))))
    def draw(self):
        w=max(10,self.canvas.winfo_width());h=max(10,self.canvas.winfo_height());self.canvas.delete('all')
        if self.before is None:
            self.canvas.create_text(w/2,h/2,text='添加照片后在这里预览\n左侧选择预设，右侧调整和导出',fill='#c8d2cd',font=('Microsoft YaHei UI',12),justify='center')
            return
        before=self.before.copy();before.thumbnail((max(1,w-20),max(1,h-20)),Image.Resampling.LANCZOS)
        after=(self.after or self.before).resize(before.size,Image.Resampling.LANCZOS)
        mode=self.view.get();out=before.copy() if mode=='before' else after.copy()
        if mode=='split':
            cut=round(out.width*self.split);out.paste(before.crop((0,0,cut,out.height)),(0,0))
        self.display_photo=ImageTk.PhotoImage(out)
        x=(w-out.width)//2;y=(h-out.height)//2;self.image_box=(x,y,out.width,out.height)
        self.canvas.create_image(x,y,image=self.display_photo,anchor='nw')
        if mode=='split':
            edge=x+out.width*self.split;self.canvas.create_line(edge,y,edge,y+out.height,fill='white',width=2)
            self.canvas.create_oval(edge-9,h/2-15,edge+9,h/2+15,fill='white',outline='#17644e')
        labels=[(x+12,'原图')] if mode=='before' else [(x+12,'处理后')]
        if mode=='split':labels=[(x+12,'原图'),(x+out.width-60,'处理后')]
        for lx,label in labels:
            self.canvas.create_rectangle(lx-4,y+8,lx+48,y+32,fill='#242a28',outline='')
            self.canvas.create_text(lx,y+20,text=label,anchor='w',fill='white',font=('Microsoft YaHei UI',9))
    def drag(self,event):
        if self.view.get()=='split' and hasattr(self,'image_box'):
            x,_,width,_=self.image_box;self.split=min(.98,max(.02,(event.x-x)/width));self.draw()
    def export_changed(self):
        fmt=FORMAT_CHOICES[self.format.get()]
        formats={resolve_format(p,fmt) for p in self.files} if self.files else ({fmt} if fmt!='auto' else {'jpeg','tiff','png'})
        self.quality_spin.configure(state='normal' if 'jpeg' in formats else 'disabled')
        self.bits_combo.configure(state='readonly' if formats & {'tiff','png'} else 'disabled')
        prefix='RAW → TIFF\n' if fmt=='auto' else ''
        if formats=={'jpeg'}:hint='JPEG  8 位'
        elif formats=={'webp'}:hint='WebP  8 位无损'
        elif formats=={'bmp'}:hint='BMP  8 位'
        elif formats <= {'tiff','png'}:hint=f'{self.bits.get()} 位无损'
        else:hint='PNG / TIFF  16 位可选'
        self.export_text.set(prefix+hint)
    def choose_output(self):
        folder=filedialog.askdirectory(title='选择输出文件夹')
        if folder:self.output.set(folder)
    def help(self):os.startfile(resource_file('使用说明.html'))
    def export_luts(self):
        folder=filedialog.askdirectory(title='选择保存 LUT 和 Lightroom 配置的文件夹')
        if not folder:return
        try:
            dest=Path(folder)/('DCU5_DataTransfer_'+VERSION)
            files=export_resources(dest)
            messagebox.showinfo('导出完成',f'已保存 {len(files)} 个文件到：\n{dest}')
        except Exception as e:messagebox.showerror('导出失败',str(e))
    def open_output(self):
        folder=Path(self.output.get());folder.mkdir(parents=True,exist_ok=True);os.startfile(folder)
    def install(self):
        try:
            target,count=install_profiles();messagebox.showinfo('配置文件已复制',f'{count} 个配置文件已放入：\n{target}\n重启 Lightroom / Camera Raw 后查看配置文件。')
        except Exception as e:messagebox.showerror('安装未完成',str(e))
    def start(self):
        if self.busy:return
        if not self.files:messagebox.showinfo('添加照片','请先添加照片。');return
        try:
            quality=self.quality.get();bits=int(self.bits.get())
            if not 80<=quality<=100:raise ValueError()
            if not self.output.get().strip():raise ValueError()
        except Exception:messagebox.showerror('导出设置','JPEG 质量请填写 80–100，并选择输出文件夹。');return
        options=(list(self.files),Path(self.output.get()),self.look.get(),self.amount.get()/100,self.exposure.get(),FORMAT_CHOICES[self.format.get()],quality,bits)
        self.busy=True;self.run_button.configure(state='disabled');self.video_run.configure(state='disabled');self.progress['value']=0
        self.status.set('正在导出原尺寸照片…')
        threading.Thread(target=self.worker,args=options,daemon=True).start()
    def worker(self,files,folder,look,amount,ev,fmt,quality,bits):
        success=0;errors=[];outputs=[]
        for i,path in enumerate(files):
            try:
                dest=destination_for(path,folder,look,fmt)
                def progress(v):self.events.put(('progress',((i+v)/len(files)*100,f'正在导出 {i+1}/{len(files)}：{Path(path).name}')))
                report=process(path,dest,look,amount,ev,progress=progress,jpeg_quality=quality,bit_depth=bits)
                outputs.append({'file':str(dest),'format':report['output_format'],'bits':report['output_bits']});success+=1
            except Exception as e:errors.append(Path(path).name+'：'+str(e))
        self.events.put(('done',(success,errors,outputs)))
    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='loaded':
                    token,data=value
                    if token!=self.generation:continue
                    self.preview_data=data;rgb,linear,alpha,size,bits=data
                    self.before=to_image(srgb_encode(rgb) if linear else rgb,alpha);self.after=None
                    self.file_info.set(f'{Path(self.current_path).name} · {size[0]} × {size[1]} · {bits} 位'+(' · RAW 预览解码，导出为完整尺寸' if Path(self.current_path).suffix.lower() in RAW_EXT else ''))
                    self.request_render();self.draw()
                elif kind=='thumbs':
                    token,thumbs=value
                    if token!=self.generation:continue
                    self.thumbnails=thumbs;self.build_cards()
                elif kind=='rendered':
                    token,rid,after=value
                    if token==self.generation and rid==self.render_id:self.after=after;self.draw()
                elif kind=='load_error':
                    token,error=value
                    if token==self.generation:self.file_info.set('预览失败：'+error);self.status.set(error)
                elif kind=='progress':
                    percent,text=value;self.progress['value']=percent;self.status.set(text)
                elif kind=='done':
                    success,errors,outputs=value;self.busy=False;self.run_button.configure(state='normal');self.video_run.configure(state='normal')
                    self.progress['value']=100;self.status.set(f'已导出 {success} 个文件；失败 {len(errors)} 个。原片已保留。')
                    if errors:messagebox.showwarning('部分文件未导出','\n'.join(errors[:12]))
                    self.last_outputs=outputs
        except queue.Empty:pass
        self.root.after(80,self.poll)
    def make_video_tab(self,frame):
        self.video_files=[];self.video_selection=tk.StringVar(value='尚未选择视频')
        self.video_look=tk.StringVar(value='GOLD');self.video_codec=tk.StringVar(value='ProRes HQ (MOV, 10 bit)')
        self.video_ready=tk.BooleanVar(value=False)
        ttk.Label(frame,text='视频导出',style='Title.TLabel').pack(anchor='w')
        ttk.Label(frame,text='输入应已转换为 Rec.709 / Gamma 2.4。照片的浓淡与曝光滑块不用于此视频流程。',wraplength=850).pack(anchor='w',pady=14)
        ttk.Button(frame,text='选择视频',command=self.choose_video).pack(anchor='w')
        ttk.Label(frame,textvariable=self.video_selection).pack(anchor='w',pady=12)
        ttk.Label(frame,text='风格').pack(anchor='w')
        ttk.Combobox(frame,textvariable=self.video_look,values=list(LOOKS),state='readonly',width=35).pack(anchor='w',pady=8)
        ttk.Label(frame,text='输出编码').pack(anchor='w')
        ttk.Combobox(frame,textvariable=self.video_codec,values=['ProRes HQ (MOV, 10 bit)','HEVC (MP4, 10 bit)'],state='readonly',width=35).pack(anchor='w',pady=8)
        ttk.Label(frame,text='ProRes 适合继续剪辑，文件较大；HEVC 文件较小。两者都经过重新编码。').pack(anchor='w',pady=8)
        ttk.Checkbutton(frame,text='我确认输入是 Rec.709 / Gamma 2.4，已完成 Log / HDR 转换',variable=self.video_ready).pack(anchor='w',pady=14)
        ttk.Entry(frame,textvariable=self.output,width=80).pack(anchor='w')
        ttk.Button(frame,text='选择输出文件夹',command=self.choose_output).pack(anchor='w',pady=8)
        self.video_run=ttk.Button(frame,text='导出视频',command=self.start_video,style='Accent.TButton');self.video_run.pack(anchor='w',pady=12)
        ttk.Label(frame,text='已内置视频编码器。视频风格按标准强度应用；通过顶部“导出 LUT / 配置”可将 LUT 保存给剪辑软件使用。',wraplength=850,style='Hint.TLabel').pack(anchor='w',pady=8)
    def choose_video(self):
        if self.busy:return
        paths=filedialog.askopenfilenames(title='选择视频',filetypes=[('视频','*.mov *.mp4 *.mxf *.mkv')])
        if paths:self.video_files=list(paths);self.video_selection.set(f'{len(paths)} 个视频 · '+Path(paths[0]).name)
    def start_video(self):
        if self.busy:return
        if not self.video_files or not self.video_ready.get():
            messagebox.showinfo('视频输入','选择视频，并确认输入色彩空间。');return
        self.busy=True;self.run_button.configure(state='disabled');self.video_run.configure(state='disabled')
        codec='prores' if self.video_codec.get().startswith('ProRes') else 'hevc10'
        threading.Thread(target=self.video_worker,args=(list(self.video_files),Path(self.output.get()),self.video_look.get(),codec),daemon=True).start()
    def video_worker(self,files,folder,look,codec):
        success=0;errors=[];outputs=[]
        for i,path in enumerate(files):
            try:
                ext='.mov' if codec=='prores' else '.mp4';stem=Path(path).stem+'_'+look;dest=folder/(stem+ext);n=1
                while dest.exists():dest=folder/(stem+f'_{n}'+ext);n+=1
                self.events.put(('progress',(i/len(files)*100,f'正在导出视频 {i+1}/{len(files)}')))
                render(path,dest,look,codec=codec);success+=1;outputs.append({'file':str(dest)})
            except Exception as e:errors.append(Path(path).name+'：'+str(e))
        self.events.put(('done',(success,errors,outputs)))
    def close(self):
        if self.busy:messagebox.showinfo('正在处理','请等导出完成后关闭。');return
        self.generation+=1;self.pool.shutdown(wait=False,cancel_futures=True);self.root.destroy()

def main():
    if '--self-test' in sys.argv:
        import rawpy,tifffile,imagecodecs
        checks=[]
        for key in LOOKS:
            out=apply(np.array([[0.,0.,0.],[.5,.25,.75],[1.,1.,1.]]),key)
            assert out.shape==(3,3) and np.isfinite(out).all();checks.append(key)
        (report_dir()/'selftest.json').write_text(json.dumps({'version':VERSION,'looks':len(LOOKS),'frozen':bool(getattr(sys,'frozen',False)),
            'checks':checks,'rawpy':rawpy.__version__,'tifffile':tifffile.__version__,'imagecodecs':imagecodecs.__version__,
            'bundled_ffmpeg':Path(ffmpeg_path()).is_file(),'export_assets':len(resource_names('exports'))},indent=2),encoding='utf-8');return
    if '--verify-portable' in sys.argv:
        from verify_portable_workflow import main as verify_portable
        i=sys.argv.index('--verify-portable');verify_portable(Path(sys.argv[i+1]));return
    if '--render-photo' in sys.argv:
        i=sys.argv.index('--render-photo');process(sys.argv[i+1],sys.argv[i+2],sys.argv[i+3] if len(sys.argv)>i+3 else 'GOLD');return
    if '--verify-workflow' in sys.argv:
        from verify_photo_workflow import main as verify_main
        i=sys.argv.index('--verify-workflow');verify_main(Path(sys.argv[i+1]));return
    if '--ui-check' in sys.argv:
        root=tk.Tk();root.withdraw();app=App(root);root.update_idletasks()
        assert len(app.cards)==11 and app.format.get()=='跟随输入格式'
        app.group.set('黑白');app.build_cards();assert len(app.cards)==9
        app.select_preset('Portrait');app.set_amount(50);app.exposure.set(1);app.settings_changed()
        assert '50%' in app.amount_text.get() and app.look.get()=='Portrait'
        assert 'JPEG' in app.export_text.get()
        app.group.set('彩色');app.build_cards();app.reset()
        if len(sys.argv)>sys.argv.index('--ui-check')+1:
            import time
            sample=Path(sys.argv[sys.argv.index('--ui-check')+1]);app.add_files([str(sample)])
            deadline=time.monotonic()+30
            while time.monotonic()<deadline and (app.after is None or len(app.thumbnails)!=28):
                root.update();time.sleep(.03)
            assert app.after is not None and len(app.thumbnails)==28,app.file_info.get()
            app.view.set('split');app.draw();app.format.set('PNG');app.export_changed()
            assert str(app.bits_combo['state'])=='readonly' and str(app.quality_spin['state'])=='disabled'
            app.format.set('JPEG');app.export_changed()
            assert str(app.bits_combo['state'])=='disabled' and str(app.quality_spin['state'])=='normal'
            app.format.set('跟随输入格式');app.export_changed()
        result={'version':VERSION,'hidden_tk_controls':True,'visible_gui_tested':False,'default_format':app.format.get(),
            'preset_cards':sum(1 for _ in LOOKS),'tabs':len(app.tabs.tabs())}
        (report_dir()/'ui-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        app.close();return
    root=tk.Tk();App(root);root.mainloop()

if __name__=='__main__':
    try:main()
    except Exception:
        error_file=report_dir()/'error.log'
        error_file.write_text(traceback.format_exc(),encoding='utf-8')
        if not any(x.startswith('--') for x in sys.argv[1:]):messagebox.showerror('启动失败','详情已写入：\n'+str(error_file))
        sys.exit(1)
