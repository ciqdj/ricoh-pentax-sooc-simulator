"""Trackless scroll thumbs and pointer-anchored full-resolution inspection."""
from concurrent.futures import ThreadPoolExecutor
import math,time
import tkinter as tk
import numpy as np
from PIL import Image
from native_backend import RenderCancelled
from postprocess import process
from basic_adjustments import halo as basic_halo


class SlimScroll(tk.Canvas):
    def __init__(self,parent,command,bg='#1c1c1e'):
        super().__init__(parent,width=12,bg=bg,bd=0,highlightthickness=0,takefocus=False)
        self.command=command;self.first=0;self.last=1;self.hot=False;self.anchor=None
        self.bind('<Configure>',lambda e:self.redraw())
        self.bind('<Enter>',lambda e:self.hover(True));self.bind('<Leave>',lambda e:self.hover(False))
        self.bind('<Button-1>',self.down);self.bind('<B1-Motion>',self.drag)
        self.bind('<ButtonRelease-1>',self.release)
        self.bind('<MouseWheel>',lambda e:(self.command('scroll',-int(e.delta/120 or (1 if e.delta>0 else -1))*3,'units'),'break')[-1])

    def geometry_values(self):
        h=max(1,self.winfo_height());length=min(h,max(30,h*(self.last-self.first)))
        track=max(0,h-length);span=max(1e-9,1-(self.last-self.first));top=track*self.first/span
        return h,length,track,span,top

    def set(self,first,last):self.first=float(first);self.last=float(last);self.redraw()

    def hover(self,value):self.hot=value;self.redraw()

    def redraw(self):
        self.delete('all')
        if self.last-self.first>=.999:return
        h,length,track,span,top=self.geometry_values();width=6;cx=6
        # A single rounded line: no rectangular trough, border, or arrows.
        self.create_line(cx,top+width/2,cx,top+length-width/2,width=width,capstyle='round',fill='#ffffff' if self.hot or self.anchor is not None else '#aaaab2',tags='thumb')

    def release(self,event=None):self.anchor=None;self.redraw()

    def down(self,e):
        h,length,track,span,top=self.geometry_values()
        self.anchor=(e.y-top) if top<=e.y<=top+length else length/2
        self.drag(e)

    def drag(self,e):
        if self.anchor is None:return
        h,length,track,span,top=self.geometry_values()
        self.command('moveto',max(0,min(span,(e.y-self.anchor)/max(1,track)*span)));self.redraw()


class PhotoViewport:
    def __init__(self,app):
        self.app=app;self.scale=0.;self.center=None;self.dragging=None;self.serial=0
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='DetailView');self.future=None;self.request=None
        self.last_key=None;self.last_pair=None;self.closed=False;self.ready_key=None
        self.reference_size=None;self.source_size=None;self.running_key=None;self.running_request=None
        self.interaction_until=0.;self.quick_key=None;self.quick_pair=None
        self.ready_interim=False;self.interim_done=None;self.interactive_limit=480

    def reset(self):
        self.scale=0.;self.center=None;self.last_key=None;self.last_pair=None;self.ready_key=None;self.serial+=1;self.request=None
        self.app.want_detail=False
        self.reference_size=None;self.source_size=None;self.dragging=None;self.interaction_until=0.
        self.quick_key=None;self.quick_pair=None
        self.ready_interim=False;self.interim_done=None

    def size(self):
        a=self.app;report=a.factory.get(a.look.get(),{})
        if a.detail_cache and a.detail_cache[:2]==(a.generation,a.requested):
            rgb=a.detail_cache[2];self.source_size=(rgb.shape[1],rgb.shape[0])
        elif report.get('native_width',0)>0 and report.get('native_height',0)>0:
            self.source_size=(report['native_width'],report['native_height'])
        # Rendering another look may not have its report yet. Keep this
        # photo's coordinate system; a smaller sample is not a smaller photo.
        return self.source_size or (a.before.size if a.before else (1,1))

    def canvas_size(self):
        return max(10,self.app.canvas.winfo_width()-20),max(10,self.app.canvas.winfo_height()-20)

    def fit_scale(self):
        w,h=self.size();cw,ch=self.canvas_size();return min(1,cw/w,ch/h)

    def clamp_center(self):
        w,h=self.size();cw,ch=self.canvas_size();s=self.scale or self.fit_scale()
        x,y=self.center or (w/2,h/2);hx=min(w/2,cw/(2*s));hy=min(h/2,ch/(2*s))
        self.center=(min(w-hx,max(hx,x)),min(h-hy,max(hy,y)))

    def zoom(self,new_scale,x=None,y=None):
        if self.app.before is None:return
        cw,ch=self.canvas_size();w,h=self.size();old=self.scale or self.fit_scale();cx,cy=self.center or (w/2,h/2)
        x=cw/2+10 if x is None else x;y=ch/2+10 if y is None else y
        new=max(self.fit_scale(),min(8,new_scale));px=cx+(x-10-cw/2)/old;py=cy+(y-10-ch/2)/old
        self.scale=0 if new<=self.fit_scale()*1.001 else new
        self.center=(px-(x-10-cw/2)/new,py-(y-10-ch/2)/new);self.clamp_center()
        self.reference_size=(w,h)
        self.interaction_until=time.monotonic()+.07
        self.app.want_detail=self.scale>0;self.app.draw()

    def wheel(self,e):
        if not e.delta:return 'break'
        old=self.scale or self.fit_scale();self.zoom(old*(1.25 if e.delta>0 else .8),e.x,e.y);return 'break'

    def fit(self):
        size=self.size();self.reset();self.source_size=size;self.app.draw()
    def actual(self):self.zoom(1.)

    def down(self,e):
        a=self.app
        if a.after is not None and a.view.get()=='split' and hasattr(a,'box') and abs(e.x-(a.box[0]+a.box[2]*a.split))<18:
            self.dragging=('split',);self.drag(e)
        elif self.scale:
            self.clamp_center();self.dragging=('pan',e.x,e.y,self.center)
        else:self.dragging=None

    def drag(self,e):
        if not self.dragging:return
        a=self.app
        if self.dragging[0]=='split':a.split=max(.02,min(.98,(e.x-a.box[0])/a.box[2]));a.draw();return
        _,x,y,center=self.dragging;self.center=(center[0]-(e.x-x)/self.scale,center[1]-(e.y-y)/self.scale)
        self.interaction_until=time.monotonic()+.07
        self.clamp_center();a.draw()

    def pointer(self,e):
        a=self.app;over=a.after is not None and a.view.get()=='split' and hasattr(a,'box') and abs(e.x-(a.box[0]+a.box[2]*a.split))<18
        a.canvas.configure(cursor='sb_h_double_arrow' if over else 'fleur' if self.scale else '')

    @staticmethod
    def crop_geometry(size,scale,center,canvas):
        """The same integer source rectangle for fast display and exact detail."""
        w,h=size;cw,ch=canvas
        width=min(w,max(1,round(cw/scale)));height=min(h,max(1,round(ch/scale)))
        x=max(0,min(w-width,round(center[0]-width/2)));y=max(0,min(h-height,round(center[1]-height/2)))
        return (x,y,x+width,y+height),(max(1,round(width*scale)),max(1,round(height*scale)))

    @staticmethod
    def same_context(first,second):
        return (first is not None and second is not None and len(first)==7 and len(second)==7
                and first[:2]==second[:2] and first[5:]==second[5:])

    @staticmethod
    def same_source(first,second):
        return (first is not None and second is not None and len(first)==7 and len(second)==7
                and first[:2]==second[:2] and first[6]==second[6])

    @classmethod
    def same_geometry(cls,first,second):
        return cls.same_source(first,second) and first[2:5]==second[2:5]

    def quick_frame(self,key):
        """Move cached image pixels now; never run a colour operation on input events.

        The fit previews cover newly exposed edges. Reproject the overlapping
        part of the last exact detail crop on top, retaining its sharpness.
        This frame is temporary; ready_key is reserved for the exact worker.
        """
        a=self.app
        fit_revision=getattr(a,'_post_presented',-1)
        quick_key=(key,id(a.before),id(a.after),id(self.last_pair),self.ready_key,fit_revision)
        if self.quick_key==quick_key:return self.quick_pair
        w,h=self.size();box,target=self.crop_geometry((w,h),key[2],key[3],key[4])
        x,y,x2,y2=box;tw,th=target
        def project(image,rect=(0,0,None,None)):
            if image is None:return None
            l,t,r,b=rect;r=tw if r is None else r;b=th if b is None else b
            region=((x+l*(x2-x)/tw)*image.width/w,(y+t*(y2-y)/th)*image.height/h,
                    (x+r*(x2-x)/tw)*image.width/w,(y+b*(y2-y)/th)*image.height/h)
            return image.transform((r-l,b-t),Image.Transform.EXTENT,region,Image.Resampling.BILINEAR)
        pair=None
        # A completed intermediate detail frame is useful while the next one
        # runs. Never paste its older colours over a newer fit-preview result.
        if (self.last_pair is not None and self.same_source(self.ready_key,key)
                and self.ready_key[5]>=fit_revision):
            old_box,_=self.crop_geometry((w,h),self.ready_key[2],self.ready_key[3],self.ready_key[4])
            ox,oy,ox2,oy2=old_box
            left=max(x,ox);top=max(y,oy);right=min(x2,ox2);bottom=min(y2,oy2)
            # Keep the pasted rectangle wholly inside the old detail crop.
            # Uncovered areas already contain the current full-image preview.
            l=math.ceil((left-x)*tw/(x2-x));t=math.ceil((top-y)*th/(y2-y))
            r=math.floor((right-x)*tw/(x2-x));b=math.floor((bottom-y)*th/(y2-y))
            if l<r and t<b:
                pair=[]
                for source,image in zip((a.before,a.after),self.last_pair):
                    if source is None:pair.append(None);continue
                    frame=Image.new(source.mode,target)
                    # Reproject only newly exposed strips from the fit image;
                    # do not resample an entire frame and overwrite it again.
                    for strip in ((0,0,l,th),(r,0,tw,th),(l,0,r,t),(l,b,r,th)):
                        if strip[0]<strip[2] and strip[1]<strip[3]:
                            frame.paste(project(source,strip),strip[:2])
                    dx=(ox-x)*tw/(x2-x);dy=(oy-y)*th/(y2-y)
                    if (tw/(x2-x)==image.width/(ox2-ox) and th/(y2-y)==image.height/(oy2-oy)
                            and abs(dx-round(dx))<1e-9 and abs(dy-round(dy))<1e-9):
                        # Ordinary 100%/200% pans are exact integer copies.
                        patch=image.crop((l-round(dx),t-round(dy),r-round(dx),b-round(dy)))
                    else:
                        region=((x+l*(x2-x)/tw-ox)*image.width/(ox2-ox),
                                (y+t*(y2-y)/th-oy)*image.height/(oy2-oy),
                                (x+r*(x2-x)/tw-ox)*image.width/(ox2-ox),
                                (y+b*(y2-y)/th-oy)*image.height/(oy2-oy))
                        patch=image.transform((r-l,b-t),Image.Transform.EXTENT,region,Image.Resampling.BILINEAR)
                    frame.paste(patch,(l,t));pair.append(frame)
        if pair is None:pair=[project(a.before),project(a.after)]
        self.quick_key=quick_key;self.quick_pair=tuple(pair)
        return self.quick_pair

    def parameters_changed(self):
        """Coalesce colour edits without repainting unchanged geometry."""
        a=self.app
        if not self.scale or a.before is None:return None
        cache=a.detail_cache
        if cache is not None and cache[:2]!=(a.generation,a.requested):cache=None
        size=self.size()
        if self.reference_size is not None and size!=self.reference_size:
            if self.center:self.center=(self.center[0]*size[0]/self.reference_size[0],self.center[1]*size[1]/self.reference_size[1])
            self.reference_size=size
        self.clamp_center();key=(a.generation,a.requested,self.scale,self.center,self.canvas_size(),a.post_revision,id(cache))
        if self.ready_key is not None and not self.same_source(self.ready_key,key):
            self.ready_key=None;self.last_pair=None;self.ready_interim=False
        if key!=self.last_key:
            self.last_key=key;self.serial+=1
            settings=a.render_post_settings() if hasattr(a,'render_post_settings') else a.current_post()
            # Source AI is already upstream of the filter in detail_cache[3].
            # Replacing this image with a raw clean array loses the filter.
            balance=a.render_balance() if hasattr(a,'render_balance') else a.current_balance()
            self.request=None if cache is None else (self.serial,key,cache,balance,dict(settings,denoise=0))
            if self.future is None:self.submit()
        return key

    def draw(self,draw_pair):
        a=self.app
        a.zoom_text.set('适应窗口' if not self.scale else f'{self.scale*100:.0f}%')
        if not self.scale or a.before is None:
            if self.last_key is not None or self.request is not None:
                self.serial+=1;self.last_key=None;self.ready_key=None;self.last_pair=None;self.request=None
                self.quick_key=None;self.quick_pair=None;self.ready_interim=False
            return draw_pair()
        key=self.parameters_changed()
        pair=self.last_pair if key==self.ready_key and self.last_pair is not None else self.quick_frame(key)
        before,after=a.before,a.after
        try:a.before,a.after=pair;draw_pair()
        finally:a.before,a.after=before,after
        if key!=self.ready_key or self.ready_interim:
            text='正在准备原尺寸细节…' if a.detail_cache is None or a.detail_cache[:2]!=(a.generation,a.requested) else '更新细节…'
            a.canvas.create_text(16,44,text=text,anchor='nw',fill='#ffffff',tags='detail-loading')

    def submit(self):
        # Continuous wheel/pan events update only the latest pending request.
        # A short idle interval starts one exact job, instead of cancelling and
        # restarting expensive colour work for every pointer position.
        if self.request is None or self.closed or time.monotonic()<self.interaction_until:return
        request=self.request
        serial,key,cache,balance,settings=request
        interacting=time.monotonic()<getattr(self.app,'_post_interacting_until',0.)
        if interacting and self.interim_done==key:return
        self.request=None;self.running_key=key
        _,_,scale,center,(cw,ch),_,_=key
        def cancelled():
            # New colour values for the same source and geometry replace only
            # the pending request. Let an ordered intermediate frame complete.
            return (self.closed or self.app.generation!=key[0]
                    or not self.same_geometry(self.last_key,key))
        limit=self.interactive_limit
        def run():
            started=time.perf_counter()
            if cancelled():raise RenderCancelled()
            _,_,base,after,alpha=cache;h,w=base.shape[:2]
            (x,y,x2,y2),output_size=self.crop_geometry((w,h),scale,center,(cw,ch))
            width=x2-x;height=y2-y
            # Spatial edits need original neighbouring pixels, irrespective
            # of display zoom. Process the full finite support, then crop.
            margin=basic_halo(settings.get('basic'),scale=1.0)
            left=max(0,x-margin);top=max(0,y-margin)
            right=min(w,x2+margin);bottom=min(h,y2+margin)
            rgb=after[top:bottom,left:right]
            # Inspect exactly the same full-image result that export uses.
            # Re-running a denoiser on the viewport would use different image
            # context and could change the result every time the user pans.
            patch_alpha=None if alpha is None else alpha[top:bottom,left:right]
            interim=interacting and max(width,height)>limit
            pixel_scale=1.
            sx=x-left;sy=y-top;sw=width;sh=height
            if interim:
                # Only a temporary display sample. Keep uint16 values and the
                # full original-pixel path unchanged for the settled frame.
                factor=limit/max(width,height)
                rows=max(1,round(rgb.shape[0]*factor));cols=max(1,round(rgb.shape[1]*factor))
                iy=np.minimum(rgb.shape[0]-1,((np.arange(rows)+.5)*rgb.shape[0]/rows).astype(np.intp))
                ix=np.minimum(rgb.shape[1]-1,((np.arange(cols)+.5)*rgb.shape[1]/cols).astype(np.intp))
                ry=rows/rgb.shape[0];rx=cols/rgb.shape[1]
                rgb=rgb[iy[:,None],ix[None,:]]
                if patch_alpha is not None:patch_alpha=patch_alpha[iy[:,None],ix[None,:]]
                sx=max(0,min(cols-1,round(sx*rx)));sy=max(0,min(rows-1,round(sy*ry)))
                sw=max(1,min(cols-sx,round(width*rx)));sh=max(1,min(rows-sy,round(height*ry)))
                pixel_scale=min(rx,ry)
            rgb=process(rgb,balance,settings,cancelled,scale=pixel_scale,alpha=patch_alpha)
            rgb=rgb[sy:sy+sh,sx:sx+sw]
            processed_alpha=None if patch_alpha is None else patch_alpha[sy:sy+sh,sx:sx+sw]
            source=base[y:y+height,x:x+width]
            def convert(array,channel):
                a=((array.astype(np.uint32)+128)//257).astype(np.uint8)
                if channel is not None:
                    a=np.rint(a*channel[...,None]+240*(1-channel[...,None])).astype(np.uint8)
                im=Image.fromarray(a)
                return im.resize(output_size,Image.Resampling.LANCZOS) if im.size!=output_size else im
            result=(convert(source,None if alpha is None else alpha[y:y+height,x:x+width]),convert(rgb,processed_alpha))
            if cancelled():raise RenderCancelled()
            return key,result,interim,time.perf_counter()-started
        self.running_request=request
        self.future=self.pool.submit(run)

    def poll(self):
        updated=False
        if self.future is not None and self.future.done():
            f=self.future;running=self.running_key;request=self.running_request
            self.future=None;self.running_key=None;self.running_request=None
            try:
                key,result,interim,elapsed=f.result()
                if (self.same_geometry(key,self.last_key)
                        and (self.ready_key is None or key[5]>=self.ready_key[5])):
                    self.last_pair=result;self.ready_key=key;self.ready_interim=interim;updated=True
                    if interim:
                        self.interim_done=key
                        if elapsed>.16:self.interactive_limit=max(160,int(self.interactive_limit*.75)//16*16)
                        elif elapsed<.055:self.interactive_limit=min(640,int(self.interactive_limit*1.25)//16*16)
                        if key==self.last_key and self.request is None:self.request=request
            except RenderCancelled:pass
            except Exception as exc:
                if running==self.last_key and running and running[0]==self.app.generation:self.app.status.set('细节预览失败：'+str(exc))
        if self.future is None and self.request is not None:self.submit()
        if updated:self.app.draw()
        elif self.scale and self.future is None:
            cache=self.app.detail_cache
            # Full-resolution preparation can finish without a fit-preview
            # event. Notice that arrival once; do not recreate a Tk bitmap on
            # every poll while the already-rendered 100% view is stationary.
            if cache is not None and cache[:2]==(self.app.generation,self.app.requested) and (self.last_key is None or self.last_key[-1]!=id(cache)):
                self.app.draw()

    def close(self):
        self.closed=True;self.serial+=1;self.pool.shutdown(wait=True,cancel_futures=True)
