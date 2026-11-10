"""Custom CUBE library controls and a photo-only background render route."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue,threading,time
from ui_language import tk, ttk, filedialog, simpledialog, messagebox
import numpy as np
from filter_catalog import CUSTOM,catalog,groups,renderer_available
from native_backend import RenderCancelled
from native_app import preview


def _import_files(library,paths,brand,category,cancelled):
    records=[];errors=[]
    for path in paths:
        if cancelled():break
        try:records.append(library.import_cube(path,brand,category,cancelled=cancelled))
        except RenderCancelled:break
        except Exception as exc:errors.append(Path(path).name+'：'+str(exc))
    return records,errors


class CustomLutMixin:
    def init_custom_lut(self,root):
        from custom_lut import get_library
        self.custom_library=get_library()
        self.custom_brand_id=next(iter(self.custom_library.brands()),None)
        self.custom_brand_name=tk.StringVar(master=root,value='')
        self.custom_brand_host=None;self.custom_brand_combo=None;self.custom_manage=None
        self.custom_category_map={};self.custom_importing=False
        self.custom_import_future=None;self.custom_import_result=[]
        self.custom_import_cancel=threading.Event()
        self.custom_import_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='CustomLutImport')

    def build_custom_controls(self,parent):
        host=ttk.Frame(parent);host.grid(row=1,column=0,columnspan=3,sticky='ew',pady=(6,0))
        host.columnconfigure(0,weight=1);self.custom_brand_host=host
        combo=ttk.Combobox(host,textvariable=self.custom_brand_name,state='readonly',width=15)
        combo.grid(row=0,column=0,sticky='ew');self.custom_brand_combo=combo
        combo.bind('<<ComboboxSelected>>',lambda e:self.custom_brand_selected())
        manage=ttk.Menubutton(host,text='管理');manage.grid(row=0,column=1,padx=(5,0))
        menu=tk.Menu(manage,tearoff=False);manage.configure(menu=menu);self.custom_manage=manage
        menu.add_command(label='新建品牌…',command=self.create_custom_brand)
        menu.add_command(label='新建分类…',command=self.create_custom_category)
        menu.add_separator();menu.add_command(label='导入 CUBE…',command=self.custom_import)
        menu.add_separator();menu.add_command(label='滤镜来源…',command=self.custom_source_info)
        self.update_custom_controls()

    def custom_source_info(self):
        note=getattr(self.custom_library,'source_note',lambda brand:'')(self.custom_brand_id)
        messagebox.showinfo('滤镜来源',note or '用户导入的 CUBE。输入色彩空间应与文件说明一致。',parent=self.root)

    def filter_groups(self,brand=None):
        brand=brand or self.brand.get()
        if brand!=CUSTOM:return groups(brand)
        records=self.custom_library.catalog(brand_id=self.custom_brand_id) if self.custom_brand_id else {}
        values={'全部':tuple(records)};mapping={}
        if self.custom_brand_id:
            for identity,name in self.custom_library.categories(self.custom_brand_id).items():
                label=name if name not in ('全部','我的预设') else '分类 · '+name
                while label in values:label+=' ·'
                mapping[label]=identity;values[label]=tuple(key for key,record in records.items() if record.get('category_id')==identity)
        self.custom_category_map=mapping
        return values

    def update_custom_controls(self):
        if self.custom_brand_host is None:return
        visible=self.brand.get()==CUSTOM
        if visible:self.custom_brand_host.grid()
        else:self.custom_brand_host.grid_remove()
        brands=self.custom_library.brands()
        if self.custom_brand_id not in brands:self.custom_brand_id=next(iter(brands),None)
        self.custom_brand_combo.configure(values=tuple(brands.values()))
        self.custom_brand_name.set(brands.get(self.custom_brand_id,'暂无品牌'))
        blocked=self.busy or self.custom_importing
        self.custom_brand_combo.configure(state='disabled' if blocked or not brands else 'readonly')
        self.custom_manage.configure(state='disabled' if blocked else 'normal')
        if visible and hasattr(self,'group_combo'):
            values=self.filter_groups(CUSTOM)
            self.group_combo.configure(values=(*values,'我的预设'),state='disabled' if blocked else 'readonly')
            if self.group.get() not in (*values,'我的预设'):self.group.set('全部')

    def custom_brand_selected(self):
        match=next((key for key,name in self.custom_library.brands().items() if name==self.custom_brand_name.get()),None)
        if match is not None:self.select_custom_brand(match)

    def select_custom_brand(self,brand_id):
        if self.busy or self.custom_importing:return
        if brand_id not in self.custom_library.brands():raise ValueError('自定义品牌不存在')
        self.custom_brand_id=brand_id
        self.switch_brand(CUSTOM,force=True)

    def create_custom_brand(self,name=None):
        if self.busy or self.custom_importing:return
        if name is None:name=simpledialog.askstring('新建品牌','品牌名称',parent=self.root)
        if name is None:return
        try:record=self.custom_library.create_brand(name)
        except Exception as exc:messagebox.showerror('无法新建品牌',str(exc),parent=self.root);return
        self.select_custom_brand(record['id']);return record

    def create_custom_category(self,name=None):
        if self.busy or self.custom_importing:return
        if getattr(self.custom_library,'is_builtin_brand',lambda brand:False)(self.custom_brand_id):
            self.status.set('内置分类不可修改，请先新建自己的品牌');return
        if not self.custom_brand_id:
            self.status.set('请先新建品牌');return
        if name is None:name=simpledialog.askstring('新建分类','分类名称',parent=self.root)
        if name is None:return
        try:record=self.custom_library.create_category(self.custom_brand_id,name)
        except Exception as exc:messagebox.showerror('无法新建分类',str(exc),parent=self.root);return
        self.update_custom_controls()
        label=next(label for label,identity in self.custom_category_map.items() if identity==record['id'])
        self.group.set(label);self.build_cards();return record

    def custom_import(self,paths=None):
        if self.busy or self.custom_importing:return
        if getattr(self.custom_library,'is_builtin_brand',lambda brand:False)(self.custom_brand_id):
            self.status.set('请先新建自己的品牌，再导入 CUBE');return
        if not self.custom_brand_id:
            self.status.set('请先新建品牌');return
        if paths is None:paths=filedialog.askopenfilenames(parent=self.root,title='导入 CUBE',filetypes=[('CUBE 滤镜','*.cube')])
        if isinstance(paths,(str,Path)):paths=(paths,)
        paths=tuple(str(Path(path)) for path in (paths or ()))
        if not paths:return
        self.custom_import_cancel=threading.Event();self.custom_importing=True
        category=self.custom_category_map.get(self.group.get());self.custom_import_result=[]
        self.custom_import_future=self.custom_import_pool.submit(_import_files,self.custom_library,paths,
            self.custom_brand_id,category,self.custom_import_cancel.is_set)
        self.status.set('正在导入 CUBE…');self.update_brand_controls()

    def poll_custom_import(self):
        future=self.custom_import_future
        if future is None or not future.done():return
        self.custom_import_future=None;self.custom_importing=False
        try:records,errors=future.result()
        except Exception as exc:records=[];errors=[str(exc)]
        self.custom_import_result=records
        if self.closed:return
        self.update_brand_controls()
        if records:
            record=records[0];self.custom_brand_id=record['brand_id']
            self.switch_brand(CUSTOM,force=True)
            self.group.set(next((label for label,identity in self.custom_category_map.items() if identity==record.get('category_id')),'全部'))
            self.select_preset(record['key']);self.build_cards()
            self.status.set(f'已导入 {len(records)} 个滤镜')
        if errors:
            self.status.set(f'导入完成，{len(errors)} 个文件失败')
            messagebox.showwarning('部分 CUBE 未导入','\n'.join(errors[:8]),parent=self.root)

    def load_custom_worker(self,path,token,evictions):
        from custom_lut import CustomSession
        try:
            session=CustomSession(path,library=self.custom_library);completed=set();failed={}
            def initial_cancelled():return self.closed or token!=self.generation
            base,alpha=session.prepare(initial_cancelled)
            h,w=base.shape[:2];ratio=min(1,1200/w,850/h)
            rows=np.minimum(h-1,((np.arange(max(1,round(h*ratio)))+.5)/ratio).astype(np.intp))
            cols=np.minimum(w-1,((np.arange(max(1,round(w*ratio)))+.5)/ratio).astype(np.intp))
            sampled=base[rows[:,None],cols[None,:]] if ratio<1 else base
            sample_alpha=None if alpha is None else alpha[rows[:,None],cols[None,:]] if ratio<1 else alpha
            before=preview(sampled,sample_alpha)
            self.events.put(('native_before',(token,self.requested,before)))
            while token==self.generation and not self.closed:
                try:
                    while True:
                        generation,old=evictions.get_nowait()
                        if generation==token:completed.discard(old)
                except queue.Empty:pass
                if self.busy or not self.requested:time.sleep(.03);continue
                wanted=self.requested;revision=self.request_revision
                need_full=self.needs_full_detail() and (self.detail_cache is None or self.detail_cache[:2]!=(token,wanted))
                if need_full:completed.discard(wanted)
                if wanted in failed and failed[wanted]!=revision:failed.pop(wanted,None)
                candidates=(wanted,)
                sig=next((item for item in dict.fromkeys(candidates) if item not in completed and item not in failed),None)
                if sig is None:time.sleep(.025);continue
                key,ev,items,amount=sig;selected=sig==self.requested
                def cancelled():
                    return (self.closed or token!=self.generation or self.busy or
                        (selected and (self.requested is None or key!=self.requested[0])) or
                        (not selected and self.requested not in completed))
                def progress(text):self.events.put(('native_stage',(token,sig,text)))
                try:
                    if selected and self.needs_full_detail():
                        rgb,_,report=session.render(key,amount,ev,cancelled,progress)
                        if sig==self.requested:self.detail_cache=(token,sig,base,rgb,alpha)
                        result=(preview(base,alpha),preview(rgb,alpha),report)
                    else:
                        sampled_base,rgb,display_alpha,report=session.render_preview(key,amount,ev,cancelled,progress,maxsize=(1200,850))
                        result=(preview(sampled_base,display_alpha),preview(rgb,display_alpha),report)
                    if cancelled():raise RenderCancelled()
                    completed.add(sig);self.events.put(('native_result',(token,sig,result)))
                    self.events.put(('native_stats',(token,{'decode_count':1,'max_parallel':1})))
                except RenderCancelled:pass
                except Exception as exc:failed[sig]=revision;self.events.put(('native_error',(token,sig,str(exc))))
        except RenderCancelled:pass
        except Exception as exc:self.events.put(('native_error',(token,None,str(exc))))
