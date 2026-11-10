"""Native-engine test edition. No automatic or subjective post-processing."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import io, json, os, queue, sys, threading, time, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from native_backend import LOOKS, SEASONS, RenderSession, RenderCancelled, save
from photo_formats import PHOTO_EXT, RAW_EXT, FORMAT_CHOICES, destination_for, resolve_format
from app_paths import report_dir, ROOT

VERSION='3.2.1 原厂引擎测试版'
NAMES={v['name']:k for k,v in LOOKS.items()}
GROUPS={'全部':(*SEASONS,*(k for k in LOOKS if k not in SEASONS)),
        '四季':SEASONS,'彩色':tuple(k for k in LOOKS if k not in SEASONS and k!='Monochrome'),
        '黑白':('Monochrome',)}
DEFAULT_OUTPUT=Path(os.environ.get('USERPROFILE',str(Path.home())))/'Pictures/PentaxNative'
_PREVIEW8=((np.arange(65536,dtype=np.uint32)+128)//257).astype(np.uint8)


def preview(rgb,alpha=None,size=(1200,850)):
    if rgb.dtype==np.uint16:rgb=_PREVIEW8[rgb]
    if alpha is not None:
        rgb=np.rint(rgb.astype(np.float64)*alpha[...,None]+240*(1-alpha[...,None])).astype(np.uint8)
    image=Image.fromarray(rgb);image.thumbnail(size,Image.Resampling.LANCZOS)
    return image


class App:
    def __init__(self,root):
        self.root=root;self.events=queue.Queue();self.pool=ThreadPoolExecutor(max_workers=1)
        self.files=[];self.path=None;self.generation=0;self.render_id=0;self.busy=False
        self.before=None;self.after=None;self.split=.5;self.cache={};self.cards={}
        self.closed=False;self.requested_key='Natural';self.preview_started=0;self.preview_stage='';self.preview_error=''
        self.preview_times={};self.preview_stats={};self.photo_started=0;self.worker_failed=False
        self.look=tk.StringVar(value=LOOKS['Natural']['name'])
        self.group=tk.StringVar(value='全部');self.preview_order=GROUPS['全部']
        self.format=tk.StringVar(value='跟随输入格式');self.bits=tk.StringVar(value='16');self.quality=tk.IntVar(value=98)
        self.output=tk.StringVar(value=str(DEFAULT_OUTPUT));self.view=tk.StringVar(value='split')
        self.status=tk.StringVar(value='添加照片，选择原厂滤镜，再导出。原文件不修改。')
        self.info=tk.StringVar(value='尚未选择照片');self.hint=tk.StringVar()
        root.title('賓得濾鏡工具 · '+VERSION);root.geometry('1280x850');root.minsize(1050,720)
        root.configure(bg='#f2f4f1');style=ttk.Style(root);style.theme_use('clam')
        style.configure('.',font=('Microsoft YaHei UI',10),background='#f2f4f1')
        style.configure('TButton',padding=7);style.configure('Hint.TLabel',foreground='#59675f')
        style.configure('Accent.TButton',background='#17644e',foreground='white')
        head=ttk.Frame(root,padding=16);head.pack(fill='x')
        ttk.Label(head,text='賓得濾鏡工具',font=('Microsoft YaHei UI',19,'bold')).pack(side='left')
        ttk.Label(head,text=VERSION,style='Hint.TLabel').pack(side='left',padx=18)
        ttk.Button(head,text='版本说明',command=self.help).pack(side='right')
        tools=ttk.Frame(root,padding=(16,0,16,10));tools.pack(fill='x')
        ttk.Button(tools,text='＋ 添加照片 / RAW',command=self.choose,style='Accent.TButton').pack(side='left')
        ttk.Button(tools,text='添加文件夹',command=self.choose_folder).pack(side='left',padx=8)
        ttk.Label(tools,text='JPEG · PNG · TIFF · WebP · BMP · 相机 RAW',style='Hint.TLabel').pack(side='right')
        body=ttk.Frame(root,padding=(16,0));body.pack(fill='both',expand=True)
        body.columnconfigure(1,weight=1);body.rowconfigure(0,weight=1)
        left=ttk.Frame(body,width=274);left.grid(row=0,column=0,sticky='nsew',padx=(0,12));left.grid_propagate(False)
        self.left_panel=left
        ttk.Label(left,text='照片列表').pack(anchor='w')
        self.file_list=tk.Listbox(left,height=4,exportselection=False,selectbackground='#17644e',font=('Microsoft YaHei UI',9))
        self.file_list.pack(fill='x',pady=(6,14));self.file_list.bind('<<ListboxSelect>>',self.file_selected)
        ttk.Label(left,text='原厂滤镜 · 点击缩略图应用',font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w',pady=5)
        groups=ttk.Combobox(left,textvariable=self.group,values=list(GROUPS),state='readonly')
        groups.pack(fill='x',pady=(0,6));groups.bind('<<ComboboxSelected>>',lambda e:self.filter_cards())
        ttk.Label(left,text='缩略图显示当前照片的实际效果；优先生成选中的滤镜。',wraplength=260,style='Hint.TLabel').pack(anchor='w',pady=(0,6))
        gridbox=ttk.Frame(left);gridbox.pack(fill='both',expand=True)
        self.preset_canvas=tk.Canvas(gridbox,width=252,bg='#f2f4f1',highlightthickness=0)
        self.preset_canvas.pack(side='left',fill='both',expand=True)
        scroll=ttk.Scrollbar(gridbox,orient='vertical',command=self.preset_canvas.yview);scroll.pack(side='right',fill='y')
        self.preset_canvas.configure(yscrollcommand=scroll.set)
        self.card_frame=ttk.Frame(self.preset_canvas)
        self.preset_canvas.create_window((0,0),window=self.card_frame,anchor='nw')
        self.card_frame.bind('<Configure>',lambda e:self.preset_canvas.configure(scrollregion=self.preset_canvas.bbox('all')))
        self.preset_canvas.bind('<MouseWheel>',self.scroll_cards)
        for key in GROUPS['全部']:
            frame=tk.Frame(self.card_frame,width=122,height=120,bg='white',highlightthickness=2,highlightbackground='#d9e0d9')
            frame.pack_propagate(False)
            title=tk.Label(frame,text=LOOKS[key]['name'],bg='white',font=('Microsoft YaHei UI',9),wraplength=112,height=2)
            title.pack(side='bottom',fill='x')
            label=tk.Label(frame,text='等待照片',width=15,height=5,bg='#e5eae5',fg='#59675f');label.pack(fill='both',expand=True)
            for widget in (frame,label,title):
                widget.bind('<Button-1>',lambda e,k=key:self.select(k));widget.bind('<MouseWheel>',self.scroll_cards)
            self.cards[key]=(frame,label)
        self.filter_cards()
        center=ttk.Frame(body);center.grid(row=0,column=1,sticky='nsew')
        views=ttk.Frame(center);views.pack(fill='x')
        for text,value in [('左右对比','split'),('处理后','after'),('输入','before')]:
            ttk.Radiobutton(views,text=text,variable=self.view,value=value,command=self.draw).pack(side='left',padx=5)
        ttk.Button(views,text='分界线居中',command=self.reset_split).pack(side='right')
        ttk.Label(center,textvariable=self.info,wraplength=570,style='Hint.TLabel').pack(anchor='w',pady=8)
        self.canvas=tk.Canvas(center,bg='#252c28',highlightthickness=0);self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.draw());self.canvas.bind('<B1-Motion>',self.drag);self.canvas.bind('<Button-1>',self.drag)
        self.canvas.bind('<Motion>',self.comparison_pointer)
        ttk.Label(center,text='拖动分界线对比；导出保留完整尺寸。RAW 输入侧优先显示相机内嵌预览。',wraplength=570,style='Hint.TLabel').pack(anchor='w',pady=8)
        right=ttk.Frame(body,width=275,padding=(14,0,0,0));right.grid(row=0,column=2,sticky='nsew');right.grid_propagate(False)
        self.right_panel=right
        ttk.Label(right,text='输出格式',font=('Microsoft YaHei UI',12,'bold')).pack(anchor='w')
        fmt=ttk.Combobox(right,textvariable=self.format,values=list(FORMAT_CHOICES),state='readonly');fmt.pack(fill='x',pady=10)
        fmt.bind('<<ComboboxSelected>>',lambda e:self.export_changed())
        row=ttk.Frame(right);row.pack(fill='x')
        ttk.Label(row,text='位深').pack(side='left');self.bits_combo=ttk.Combobox(row,textvariable=self.bits,values=['16','8'],width=4,state='readonly');self.bits_combo.pack(side='left',padx=5)
        ttk.Label(row,text='JPEG质量').pack(side='left');self.quality_spin=ttk.Spinbox(row,textvariable=self.quality,from_=80,to=100,width=4);self.quality_spin.pack(side='left',padx=5)
        ttk.Label(right,textvariable=self.hint,wraplength=245,style='Hint.TLabel').pack(anchor='w',pady=12)
        ttk.Label(right,text='保存位置').pack(anchor='w');ttk.Entry(right,textvariable=self.output).pack(fill='x',pady=6)
        ttk.Button(right,text='选择文件夹',command=self.choose_output).pack(fill='x')
        self.run_button=ttk.Button(right,text='导出列表中的全部照片',command=self.start,style='Accent.TButton');self.run_button.pack(fill='x',pady=16)
        self.progress=ttk.Progressbar(right,maximum=100);self.progress.pack(fill='x')
        ttk.Label(right,text='输出不写入作者、签名、GPS、相机序列号或源文件路径。',wraplength=245,style='Hint.TLabel').pack(anchor='w',pady=12)
        footer=ttk.Frame(root,padding=16);footer.pack(fill='x')
        ttk.Label(footer,textvariable=self.status,wraplength=1200).pack(anchor='w')
        ttk.Label(footer,text='本地运行 · 原厂引擎测试版 · 尚未完成与原版同输入同设置的逐像素验收',style='Hint.TLabel').pack(anchor='w',pady=(5,0))
        self.export_changed();root.after(80,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def filter_cards(self):
        self.preview_order=GROUPS[self.group.get()]
        for frame,_ in self.cards.values():frame.grid_forget()
        for i,key in enumerate(self.preview_order):self.cards[key][0].grid(row=i//2,column=i%2,padx=2,pady=4,sticky='n')
        self.preset_canvas.yview_moveto(0)

    def scroll_cards(self,event):
        self.preset_canvas.yview_scroll(-int(event.delta/120),'units');return 'break'

    def help(self):
        messagebox.showinfo('版本说明',
            '此版实际运行 DCU 5.10.2.2 的原厂处理代码，包含四季和常用风格。\n\n'
            '普通照片通过不改变 RGB 像素的兼容 TIFF 进入引擎。支持的宾得 PEF 直接显影；'
            '跨品牌 RAW 先由 LibRaw 解码，关闭自动提亮。K-3 III 的春红使用 K-1 II 的 RGB 兼容处理。\n\n'
            '这不是此前的固定曲线移植版，也尚未宣称与原版 GUI 的全部处理步骤验证一致。'
            '本测试版不含视频、LUT 或 Lightroom 配置输出。\n\n'
            '左侧统一选择全部风格和分类；右侧设置导出。拖动对比分界线中间的双向按钮查看差异。\n\n'
            '照片处理后保持尺寸；PNG/TIFF 可选 16 位无损编码。JPEG 会重新有损编码。')
    def choose(self):
        if not self.busy:self.add_files(filedialog.askopenfilenames(filetypes=[('照片 / RAW',' '.join('*'+x for x in sorted(PHOTO_EXT)))]))
    def choose_folder(self):
        if self.busy:return
        folder=filedialog.askdirectory()
        if folder:self.add_files([str(p) for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in PHOTO_EXT])
    def add_files(self,paths):
        for path in paths:
            path=str(Path(path).resolve())
            if path not in self.files and Path(path).suffix.lower() in PHOTO_EXT:self.files.append(path);self.file_list.insert('end',Path(path).name)
        if self.files and not self.file_list.curselection():self.file_list.selection_set(0);self.file_selected()
        self.export_changed()
    def file_selected(self,event=None):
        selected=self.file_list.curselection()
        if not selected:return
        path=self.files[selected[0]]
        if self.path==path:return
        self.path=path;self.generation+=1;self.render_id+=1;token=self.generation
        self.before=self.after=None;self.cache={};self.preview_times={};self.preview_stats={};self.worker_failed=False
        self.photo_started=time.monotonic();self.info.set(Path(path).name+' · 读取照片…')
        for _,label in self.cards.values():label.configure(image='',text='等待生成',width=15,height=5);label.image=None
        self.request_render();self.draw();self.pool.submit(self.load,path,token)
    def load(self,path,token):
        try:
            with RenderSession(path) as session:
                im=None
                if Path(path).suffix.lower() in RAW_EXT:
                    # Embedded camera preview is only the input display; it is
                    # never used as the native render/export input.
                    import rawpy
                    try:
                        with rawpy.imread(path) as raw:thumb=raw.extract_thumb()
                        if thumb.format==rawpy.ThumbFormat.JPEG:
                            with Image.open(io.BytesIO(thumb.data)) as image:im=image.convert('RGB')
                            im.thumbnail((1200,850),Image.Resampling.LANCZOS)
                        else:im=preview(thumb.data)
                    except (rawpy.LibRawNoThumbnailError,rawpy.LibRawUnsupportedThumbnailError):pass
                if im is None:im=preview(*session.rgb_input(cancelled=lambda:token!=self.generation or self.closed))
                self.events.put(('before',(token,im)))
                completed=set();failed={};pending={};max_pending=0
                def work(key,rid):
                    def cancelled():
                        return (token!=self.generation or self.closed or self.busy or
                            (self.render_id!=rid and self.requested_key!=key and self.requested_key not in completed))
                    def progress(text):self.events.put(('stage',(token,key,text)))
                    progress('开始处理')
                    rgb,alpha,report=session.render(key,progress=progress,cancelled=cancelled)
                    if cancelled():raise RenderCancelled()
                    return preview(rgb,alpha),report
                # Decode once. Give the selected look the first worker, then
                # bound background native rendering to two independent jobs.
                with ThreadPoolExecutor(max_workers=2) as renders:
                    while token==self.generation and not self.closed:
                        for future in tuple(pending):
                            if not future.done():continue
                            key,rid=pending.pop(future)
                            try:
                                im,report=future.result();completed.add(key)
                                self.events.put(('cache',(token,key,im,report)))
                                self.events.put(('stats',(token,{'input_decode_count':session.decode_count,'max_parallel_tasks':max_pending})))
                            except RenderCancelled:self.events.put(('queued',(token,key)))
                            except Exception as exc:
                                failed[key]=rid;self.events.put(('error',(token,key,str(exc))))
                        if not self.busy:
                            wanted=self.requested_key;rid=self.render_id
                            if wanted in failed and failed[wanted]!=rid:del failed[wanted]
                            active={key for key,_ in pending.values()}
                            limit=1 if wanted not in completed else 2
                            for key in dict.fromkeys((wanted,*self.preview_order)):
                                if len(pending)>=limit:break
                                if key in completed or key in failed or key in active:continue
                                pending[renders.submit(work,key,rid)]=(key,rid)
                                max_pending=max(max_pending,len(pending))
                        time.sleep(.03)
        except RenderCancelled:pass
        except Exception as exc:self.events.put(('error',(token,None,str(exc))))
    def select(self,key):self.look.set(LOOKS[key]['name']);self.request_render()
    def request_render(self):
        key=NAMES[self.look.get()]
        for k,(frame,_) in self.cards.items():frame.configure(highlightbackground='#17644e' if k==key else '#d9e0d9')
        if not self.path:return
        self.render_id+=1;self.requested_key=key;self.preview_error=''
        if key in self.cache:self.show_result(*self.cache[key]);return
        self.after=None;self.preview_started=time.monotonic();self.preview_stage='等待当前任务结束'
        self.info.set(Path(self.path).name+' · '+LOOKS[key]['name']+' · 准备处理…')
        self.status.set('正在生成选中滤镜；左侧缩略图随后生成。');self.draw()
        if self.worker_failed:
            self.worker_failed=False;self.pool.submit(self.load,self.path,self.generation)
    def show_result(self,im,report):
        self.after=im
        self.preview_started=0;self.preview_error=''
        route='PEF 原生显影' if report['input_route'].startswith('native_pef') else 'RGB 兼容输入'
        self.info.set(f'{Path(self.path).name} · {report["native_width"]} × {report["native_height"]} · {route} · {LOOKS[report["look"]]["name"]}')
        self.draw()
    def draw(self):
        w=max(10,self.canvas.winfo_width());h=max(10,self.canvas.winfo_height());self.canvas.delete('all')
        base=self.before or self.after
        if base is None:self.canvas.create_text(w/2,h/2,text='添加照片后在这里查看原厂滤镜',fill='#cbd5ce',font=('Microsoft YaHei UI',12));return
        viewport=getattr(self,'viewport',None)
        logical_size=viewport.size() if viewport is not None and not viewport.scale else base.size
        ratio=min(1,max(1,w-20)/logical_size[0],max(1,h-20)/logical_size[1])
        display_size=tuple(max(1,round(v*ratio)) for v in logical_size)
        fitted=getattr(self,'_fitted_preview',None)
        if fitted is None or fitted[0] is not base or fitted[1] is not self.after or fitted[2:5]!=(w,h,display_size):
            # Fit uses source-photo geometry, independently of the temporary
            # sampling resolution. Zoomed frames already have viewport geometry.
            before=base.resize(display_size,Image.Resampling.LANCZOS)
            after=(self.after or base).resize(display_size,Image.Resampling.LANCZOS)
            self._fitted_preview=(base,self.after,w,h,display_size,before,after)
        else:before,after=fitted[5:]
        mode=self.view.get()
        out=before.copy() if mode=='before' else after.copy()
        if mode=='split':cut=round(out.width*self.split);out.paste(before.crop((0,0,cut,out.height)),(0,0))
        self.display=ImageTk.PhotoImage(out);x=(w-out.width)//2;y=(h-out.height)//2;self.box=(x,y,out.width,out.height)
        self.canvas.create_image(x,y,image=self.display,anchor='nw')
        if mode=='split' and self.after is not None:
            edge=x+out.width*self.split;cy=y+out.height/2
            self.canvas.create_line(edge,y,edge,y+out.height,fill='white',width=2,tags='comparison-divider')
            self.canvas.create_oval(edge-15,cy-19,edge+15,cy+19,fill='white',outline='#17644e',width=2,tags='comparison-handle')
            self.canvas.create_polygon(edge-5,cy-5,edge-10,cy,edge-5,cy+5,fill='#17644e',tags='comparison-handle')
            self.canvas.create_polygon(edge+5,cy-5,edge+10,cy,edge+5,cy+5,fill='#17644e',tags='comparison-handle')
        labels=[(x+12,'输入')] if mode=='before' or self.after is None else [(x+12,'处理后')]
        if mode=='split' and self.after is not None:labels=[(x+12,'输入'),(x+out.width-64,'处理后')]
        for lx,label in labels:
            self.canvas.create_rectangle(lx-4,y+8,lx+52,y+32,fill='#252c28',outline='')
            self.canvas.create_text(lx,y+20,text=label,anchor='w',fill='white',font=('Microsoft YaHei UI',9))
        if self.after is None and mode!='before':
            self.canvas.create_rectangle(x,y+out.height-42,x+out.width,y+out.height,fill='#252c28',outline='')
            text='处理失败，请重新选择滤镜重试' if self.preview_error else '滤镜尚未完成 · 当前显示输入照片'
            self.canvas.create_text(x+out.width/2,y+out.height-21,text=text,fill='white',font=('Microsoft YaHei UI',10))
    def drag(self,event):
        if self.after is not None and hasattr(self,'box') and self.view.get()=='split':self.split=max(.02,min(.98,(event.x-self.box[0])/self.box[2]));self.draw()
    def reset_split(self):self.split=.5;self.draw()
    def comparison_pointer(self,event):
        over=self.after is not None and self.view.get()=='split' and hasattr(self,'box') and abs(event.x-(self.box[0]+self.box[2]*self.split))<18
        self.canvas.configure(cursor='sb_h_double_arrow' if over else '')
    def export_changed(self):
        fmt=FORMAT_CHOICES[self.format.get()];formats={resolve_format(p,fmt) for p in self.files} if self.files else {fmt}
        self.bits_combo.configure(state='readonly' if formats&{'tiff','png','auto'} else 'disabled')
        self.quality_spin.configure(state='normal' if formats&{'jpeg','auto'} else 'disabled')
        self.hint.set(('JPEG→JPEG，PNG→PNG，RAW→TIFF。\n' if fmt=='auto' else '')+'PNG/TIFF 可选 16 位无损；JPEG 为有损编码。')
    def choose_output(self):
        folder=filedialog.askdirectory()
        if folder:self.output.set(folder)
    def start(self):
        if self.busy or not self.files:return
        try:
            quality=int(self.quality.get());bits=int(self.bits.get())
            if not 80<=quality<=100 or not self.output.get().strip():raise ValueError()
        except Exception:messagebox.showerror('导出设置','检查输出文件夹，JPEG 质量为 80–100。');return
        self.busy=True;self.run_button.configure(state='disabled')
        args=(list(self.files),Path(self.output.get()),NAMES[self.look.get()],FORMAT_CHOICES[self.format.get()],quality,bits)
        threading.Thread(target=self.export,args=args,daemon=True).start()
    def export(self,files,folder,look,fmt,quality,bits):
        errors=[];outputs=[]
        for i,path in enumerate(files):
            self.events.put(('progress',(100*i/len(files),f'导出 {i+1}/{len(files)}：{Path(path).name}')))
            try:
                dest=destination_for(path,folder,look,fmt);save(path,dest,look,jpeg_quality=quality,bit_depth=bits);outputs.append(str(dest))
            except Exception as exc:errors.append(Path(path).name+'：'+str(exc))
        self.events.put(('done',(outputs,errors)))
    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='cache':
                    token,key,im,report=value
                    if token!=self.generation:continue
                    self.cache[key]=(im,report)
                    self.preview_times[key]=round(time.monotonic()-self.photo_started,3)
                    if key in self.cards:
                        thumb=im.copy();thumb.thumbnail((112,76),Image.Resampling.LANCZOS);photo=ImageTk.PhotoImage(thumb)
                        label=self.cards[key][1];label.configure(image=photo,text='',width=112,height=76);label.image=photo
                    if NAMES[self.look.get()]==key:self.show_result(im,report)
                    if not self.busy:
                        count=sum(k in self.cache for k in self.preview_order)
                        self.status.set(f'当前分类预览已完成 {count}/{len(self.preview_order)}；可选择风格或导出。')
                elif kind=='before':
                    token,im=value
                    if token==self.generation:self.before=im;self.draw()
                elif kind=='error':
                    token,key,error=value
                    if token!=self.generation:continue
                    if key is None:self.worker_failed=True
                    for k,(_,label) in self.cards.items():
                        if (key is None or k==key) and k not in self.cache:label.configure(text='失败 · 点击重试',image='',width=15,height=5)
                    if key is None or key==self.requested_key:
                        self.after=None;self.preview_error=error;self.preview_started=0
                        self.info.set(Path(self.path).name+' · 处理失败：'+error);self.draw()
                    if not self.busy:self.status.set('处理失败：'+error)
                elif kind=='stage':
                    token,key,text=value
                    if token!=self.generation:continue
                    if key in self.cards and key not in self.cache:self.cards[key][1].configure(text='处理中…')
                    if key==self.requested_key:self.preview_stage=text
                    if not self.busy:self.status.set(LOOKS[key]['name']+' · '+text)
                elif kind=='queued':
                    token,key=value
                    if token==self.generation and key in self.cards and key not in self.cache:self.cards[key][1].configure(text='等待生成')
                elif kind=='stats':
                    token,stats=value
                    if token==self.generation:self.preview_stats=stats
                elif kind=='progress':self.progress['value'],text=value;self.status.set(text)
                elif kind=='done':
                    outputs,errors=value;self.busy=False;self.run_button.configure(state='normal');self.progress['value']=100
                    self.status.set(f'已导出 {len(outputs)} 张，失败 {len(errors)} 张。原片未修改。')
                    if errors:messagebox.showwarning('部分照片未完成','\n'.join(errors[:10]))
        except queue.Empty:pass
        if self.preview_started and self.after is None:
            elapsed=int(time.monotonic()-self.preview_started)
            self.info.set(f'{Path(self.path).name} · {LOOKS[self.requested_key]["name"]} · {self.preview_stage} · {elapsed} 秒')
        if not self.closed:self.root.after(80,self.poll)
    def close(self):
        if self.busy:messagebox.showinfo('正在导出','请等待本次导出完成。');return
        self.closed=True;self.generation+=1;self.render_id+=1
        # Join the cancellable worker before releasing Tk variables. Otherwise
        # its last App reference can finalize Tcl on a background thread.
        self.pool.shutdown(wait=True,cancel_futures=True);self.root.destroy()


def main():
    if '--native-worker' in sys.argv:
        from native_worker import main as worker
        raise SystemExit(worker(sys.argv[sys.argv.index('--native-worker')+1]))
    if '--verify-native' in sys.argv:
        from verify_native import main as verify
        verify(Path(sys.argv[sys.argv.index('--verify-native')+1]));return
    if '--ui-check' in sys.argv:
        root=tk.Tk();root.withdraw();app=App(root);root.update_idletasks()
        assert len(app.cards)==17 and len(NAMES)==17 and app.format.get()=='跟随输入格式'
        assert all(str(frame).startswith(str(app.left_panel)) for frame,_ in app.cards.values())
        assert not any(isinstance(w,ttk.Combobox) and str(w.cget('textvariable'))==str(app.look) for w in app.right_panel.winfo_children())
        app.select('KYUSHU');assert app.look.get()=='九秋'
        app.format.set('PNG');app.export_changed();assert str(app.bits_combo['state'])=='readonly'
        app.format.set('JPEG');app.export_changed();assert str(app.bits_combo['state'])=='disabled'
        checked_photo=False
        if '--sample' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--sample')+1]);app.add_files([str(path)])
            required=tuple(LOOKS) if '--all-previews' in sys.argv else SEASONS
            deadline=time.monotonic()+180
            while time.monotonic()<deadline and (app.after is None or app.before is None or not all(k in app.cache for k in required)):
                root.update();time.sleep(.03)
            assert app.before is not None and app.after is not None and all(k in app.cache for k in required),app.status.get()
            app.view.set('split');app.draw();assert len(app.canvas.find_withtag('comparison-handle'))==3
            from types import SimpleNamespace
            app.drag(SimpleNamespace(x=app.box[0]+app.box[2]*.7));assert abs(app.split-.7)<.01
            app.reset_split();assert app.split==.5
            app.group.set('黑白');app.filter_cards();assert app.preview_order==('Monochrome',)
            checked_photo=True
        (report_dir()/'native-ui-check.json').write_text(json.dumps({'hidden_control_check':True,'visible_gui_tested':False,
            'real_photo_and_four_cards_checked':checked_photo,'season_cards':4,'native_modes':17,
            'all_17_filters_on_left':True,'comparison_handle_and_drag_checked':checked_photo,
            'real_photo_all_17_previews_checked':checked_photo and len(app.cache)==17,
            'preview_completion_seconds':app.preview_times,'input_preparation':app.preview_stats}),encoding='utf-8')
        app.close();return
    root=tk.Tk();App(root);root.mainloop()

if __name__=='__main__':
    try:main()
    except Exception:
        (report_dir()/'native-error.log').write_text(traceback.format_exc(),encoding='utf-8')
        if not any(a.startswith('--') for a in sys.argv[1:]):messagebox.showerror('运行失败','错误记录保存在应用日志文件夹。')
        raise SystemExit(1)
