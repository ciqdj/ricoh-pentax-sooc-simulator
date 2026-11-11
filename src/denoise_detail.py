"""Original-pixel denoise inspection; no preview is used as an export result."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import queue
import threading
import time
from ui_language import tk, ttk
import numpy as np
from PIL import Image, ImageTk
from native_backend import RenderCancelled,LOOKS
from native_controls import NR_CONTROLS
from filter_catalog import PENTAX
from postprocess import blend_clean,validate as validate_post
from color_balance import transform as balance_transform
from tone_tools import apply_array as tone_array
from basic_adjustments import DEFAULT as BASIC_DEFAULT,apply as basic_apply,halo as basic_halo
from color_mixer import apply as mixer_apply


def crop_bounds(shape, center, size):
    h,w=shape[:2];cw=min(w,max(1,int(size[0])));ch=min(h,max(1,int(size[1])))
    x=max(0,min(w-cw,round(float(center[0])*w-cw/2)))
    y=max(0,min(h-ch,round(float(center[1])*h-ch/2)))
    return x,y,cw,ch


def expanded_bounds(shape,bounds,margin):
    h,w=shape[:2];x,y,cw,ch=bounds
    left=max(0,x-margin);top=max(0,y-margin)
    right=min(w,x+cw+margin);bottom=min(h,y+ch+margin)
    return left,top,right-left,bottom-top


def crop_result(result,bounds,shape,margin=0):
    """Reuse a local result only when its context covers every needed pixel."""
    if result is None:return None
    cx,cy,cw,ch=result.get('context_bounds',result['bounds'])
    x,y,w,h=bounds;left,top,width,height=expanded_bounds(shape,bounds,margin)
    if not (cx<=left and cy<=top and cx+cw>=left+width and cy+ch>=top+height):return None
    source=result.get('context_source',result['source']);clean=result.get('context_clean',result['clean'])
    sx=x-cx;sy=y-cy
    return {'source':source[sy:sy+h,sx:sx+w],'clean':clean[sy:sy+h,sx:sx+w],
            'bounds':bounds,'full':result['full'],'context_source':source,
            'context_clean':clean,'context_bounds':(cx,cy,cw,ch)}


def render_patch(rgb,center,size,model,cancelled=None,progress=None):
    """Denoise an unscaled patch with surrounding original pixels.

    This is a local trial. Neural tile origins/context can differ from a full
    image; the UI switches to the actual full result as soon as it exists.
    """
    from denoise import apply
    if cancelled and cancelled():raise RenderCancelled()
    bounds=crop_bounds(rgb.shape,center,size);x,y,w,h=bounds
    # Reserve the largest supported manual-edit halo once. Moving a slider
    # later must not run a neural model again just to obtain its neighbours.
    post_margin=basic_halo(dict.fromkeys(BASIC_DEFAULT,1),scale=1.0)
    cx,cy,cw,ch=expanded_bounds(rgb.shape,bounds,post_margin)
    # The native spatial module reads seven rows. Keep extra context here;
    # neural choices retain at least their tile context on every interior edge.
    margin=16 if model=='dcu' else 96
    left=max(0,cx-margin);top=max(0,cy-margin)
    right=min(rgb.shape[1],cx+cw+margin);bottom=min(rgb.shape[0],cy+ch+margin)
    original=np.ascontiguousarray(rgb[top:bottom,left:right])
    clean=apply(original,1.,cancelled,progress,model=model)
    if cancelled and cancelled():raise RenderCancelled()
    sx=cx-left;sy=cy-top
    context={'source':original[sy:sy+ch,sx:sx+cw].copy(),
             'clean':clean[sy:sy+ch,sx:sx+cw].copy(),'bounds':(cx,cy,cw,ch),'full':False}
    return crop_result(context,bounds,rgb.shape,post_margin)


def prepare_pair(source,result,amount,alpha,bounds,settings,balance=(0,0),cancelled=None):
    """Return native-precision before/after crops, with full spatial support.

    This CPU-only helper runs on the display worker, never on the Tk thread.
    Its XY operation is the existing NumPy reference, avoiding contention with
    the inference GPU. The order and rounding match the full export pipeline.
    """
    settings=validate_post(settings)
    margin=basic_halo(settings['basic'],scale=1.0)
    left,top,width,height=expanded_bounds(source.shape,bounds,margin)
    if cancelled and cancelled():raise RenderCancelled()
    if result is None:
        original=source[top:top+height,left:left+width];clean=original
    else:
        usable=crop_result(result,bounds,source.shape,margin)
        if usable is None:raise ValueError('局部结果缺少空间调整所需的原尺寸邻域')
        cx,cy,_,_=usable['context_bounds'];sx=left-cx;sy=top-cy
        original=usable['context_source'][sy:sy+height,sx:sx+width]
        clean=blend_clean(original,usable['context_clean'][sy:sy+height,sx:sx+width],amount/100,cancelled)
    patch_alpha=None if alpha is None else alpha[top:top+height,left:left+width]
    def post(array):
        array=basic_apply(array,settings['basic'],cancelled,scale=1.0,alpha=patch_alpha)
        array=mixer_apply(array,settings['mixer'],cancelled)
        if any(balance):
            if cancelled and cancelled():raise RenderCancelled()
            maximum=np.iinfo(array.dtype).max
            array=np.floor(balance_transform(array.astype(np.float64)/maximum,*balance)*maximum+.5).astype(array.dtype)
        return tone_array(array,settings['tone'],cancelled)
    before=post(original);after=before if clean is original else post(clean)
    x,y,w,h=bounds;sx=x-left;sy=y-top
    if cancelled and cancelled():raise RenderCancelled()
    return before[sy:sy+h,sx:sx+w],after[sy:sy+h,sx:sx+w]


def render_display(source,result,amount,alpha,bounds,settings,balance,geometry,cancelled=None,processed_only=False):
    """Create a plain PIL frame without retaining a widget or invoking Tk."""
    original,clean=prepare_pair(source,result,amount,alpha,bounds,settings,balance,cancelled)
    x,y,w,h=bounds;width,height,scale=geometry
    def rgb8(array):
        values=((array.astype(np.uint32)+128)//257).astype(np.uint8) if array.dtype==np.uint16 else array
        if alpha is not None:
            a=alpha[y:y+h,x:x+w]
            if a.dtype.kind=='u':a=a.astype(np.float64)/np.iinfo(a.dtype).max
            if a.ndim==3:a=a[...,0]
            values=np.rint(values*a[...,None]+240*(1-a[...,None])).astype(np.uint8)
        return values
    dest=(max(1,round(w*scale)),max(1,round(h*scale)))
    # Pixel magnification is intentionally nearest-neighbour. The kernel was
    # already evaluated at scale=1, before any display resampling.
    left=Image.fromarray(rgb8(original)).resize(dest,Image.Resampling.NEAREST)
    right=Image.fromarray(rgb8(clean)).resize(dest,Image.Resampling.NEAREST)
    out=Image.new('RGB',(width,height),'#151517');px=(width-dest[0])//2;py=(height-dest[1])//2
    if processed_only:
        out.paste(right,(px,py))
    else:
        # Both panes show the entire SAME sensor-coordinate crop. A split
        # wipe of two different halves cannot compare a small roof/edge.
        pane=width//2
        out.paste(left,((pane-dest[0])//2,py))
        out.paste(right,(pane+1+(width-pane-1-dest[0])//2,py))
    if cancelled and cancelled():raise RenderCancelled()
    return out


def native_reference(snapshot,look,exposure,controls,amount,cancelled):
    from source_denoise_ui import SourceDenoiseMixin
    from native_backend import RenderSession,mix_amount
    with SourceDenoiseMixin.source_input(snapshot,cancelled) as path:
        with RenderSession(path) as session:
            rgb,alpha,report=session.render(look,exposure,controls=controls,cancelled=cancelled)
            if amount!=1:rgb=mix_amount(session.baseline(look,cancelled=cancelled)[0],rgb,amount)
            return rgb


def editor_reference_state(state):
    """Keep the exact manual base/look, changing only explicit native NR."""
    from photo_edit import normalize,default_state
    value=normalize(state)
    if value['denoise']['kind']!='native' or value['filter']['brand']!=PENTAX or value['filter']['look'] not in LOOKS:
        raise ValueError('原厂降噪对照需要当前启用宾得原厂降噪')
    if value['filter']['amount']!=1.:
        raise ValueError('原厂降噪对照需要原厂浓淡 100%')
    # Off means no custom NR overrides. The native engine's original/default
    # NR remains in its original position; this is not a claim of NR disabled.
    value['denoise']=default_state()['denoise']
    return value


def editor_native_reference(path,store,library,state,cancelled=None,progress=None):
    """Worker-only same-input/same-manual/same-look default-NR reference.

    No App, widget, Tk variable or callback that retains a widget is accepted.
    The current custom-NR result is separately supplied by the full-frame cache.
    """
    from editor_pipeline import EditorSession
    reference=editor_reference_state(state)
    if cancelled and cancelled():raise RenderCancelled()
    with EditorSession(path,store,library) as session:
        _,rgb,_,_=session.render(reference,maxsize=None,result=None,cancelled=cancelled,progress=progress)
        if cancelled and cancelled():raise RenderCancelled()
        return rgb


class DenoiseDetail(ttk.Frame):
    """A small split inspector. All native/model work stays off the Tk thread."""
    def __init__(self,app,parent):
        super().__init__(parent)
        self.app=app;self.pack(fill='x',pady=(4,6))
        self.center=(.5,.5);self.zoom=tk.StringVar(master=self,value='200%')
        self.note=tk.StringVar(master=self,value='')
        row=ttk.Frame(self);row.pack(fill='x')
        self.heading=tk.StringVar(master=self,value='细节预览')
        self.source_mode='native'
        # The App creates this panel before enabling its editor toolbar.
        # Presence of the initialized state identifies that construction path.
        editor=hasattr(app,'editor_state')
        selector=(ttk.Label(row,textvariable=self.heading,width=9) if editor else
                  ttk.Combobox(row,textvariable=self.heading,values=('原厂降噪','输入降噪'),state='readonly',width=12))
        selector.pack(side='left')
        if not editor:selector.bind('<<ComboboxSelected>>',lambda e:self.set_source_mode('input' if self.heading.get()=='输入降噪' else 'native'))
        control=ttk.Combobox(row,textvariable=self.zoom,values=('100%','200%','400%'),state='readonly',width=5)
        control.pack(side='right');control.bind('<<ComboboxSelected>>',lambda e:self.changed())
        self.canvas=tk.Canvas(self,width=250,height=150,bg='#151517',bd=0,highlightthickness=0,cursor='fleur')
        self.canvas.pack(fill='x',pady=4)
        self.canvas.bind('<Configure>',lambda e:self.changed())
        self.canvas.bind('<Button-1>',self.down);self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<ButtonRelease-1>',self.release)
        self.canvas.bind('<MouseWheel>',self.wheel)
        ttk.Label(self,textvariable=self.note,wraplength=252,style='Hint.TLabel').pack(fill='x')
        from ui_controls import Tooltip
        Tooltip(self.canvas,'拖动移动取样区域，滚轮缩放。\n全片处理完成后显示最终细节。')
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='DenoiseDetail')
        self.future=None;self.running=None;self.cancel_event=None;self.events=queue.SimpleQueue()
        self.display_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='DenoiseDetailDisplay')
        self.display_future=None;self.display_running=None;self.display_cancel=None
        self.display_request=None;self.display_target=None;self.display_failed=None
        self.display_error='';self.display_running_mode=None;self.display_running_serial=None
        self.display_processed_only=False
        self.closed=False;self.needs_detail=False;self.target=None;self.result=None;self.result_key=None
        self.deadline=0.;self.failed_key=None;self.last_draw=None;self.drag_origin=None;self.bitmap=None
        self._base_key=None;self._size=None;self._last_note='';self._pan_source=None;self._pan_processed_only=False;self._pan_active=False;self._spin_angle=0;self._spin_after=None;self._spin_pending=False;self._pan_painting=False;self._spin_deadline=0.
        self.native_reference=None;self.native_reference_key=None
        self.running_kind=None;self.input_preview=None;self.input_preview_key=None
        self.failed_message='';self._mode_serial=0;self._input_context_key=None;self._input_task_id=None
        self._input_revision=0
        self.placeholder('选择照片后调节降噪强度')

    def geometry_values(self):
        width=self.canvas.winfo_width();height=self.canvas.winfo_height()
        width=width if width>10 else 250;height=height if height>10 else 150
        scale=int(self.zoom.get().rstrip('%'))/100
        return width,height,scale,(max(1,int((width//2-1)/scale)),max(1,int(height/scale)))

    def set_source_mode(self,mode):
        if mode not in ('native','input'):return
        changed=self.source_mode!=mode
        self.source_mode=mode;self.heading.set('输入降噪' if mode=='input' else '原厂降噪')
        if changed:
            self._mode_serial+=1;self.changed()
            self.placeholder('正在准备输入降噪对比…' if mode=='input' else '正在准备原厂降噪对比…')
            self.note.set('');self._last_note='mode-'+mode

    def changed(self):
        self.stop();self.stop_display();self.target=None;self.deadline=time.monotonic()+.12;self.last_draw=None

    def clear_input_cache(self):
        self.changed();self._input_revision+=1;self._input_context_key=None;self._input_task_id=None
        self.input_preview=None;self.input_preview_key=None;self.failed_key=None;self.failed_message=''
        self.result=None;self.result_key=None
        self.placeholder('正在准备输入降噪对比…')

    def wheel(self,event):
        options=('100%','200%','400%');index=options.index(self.zoom.get())
        self.zoom.set(options[max(0,min(2,index+(1 if event.delta>0 else -1)))])
        self.changed();return 'break'

    def down(self,event):
        self.drag_origin=(event.x,event.y,self.center)

    def drag(self,event):
        if self.drag_origin is None or self._size is None:return
        x,y,center=self.drag_origin;_,_,scale,_=self.geometry_values();w,h=self._size
        self.center=(max(0,min(1,center[0]-(event.x-x)/scale/w)),max(0,min(1,center[1]-(event.y-y)/scale/h)))
        # AUDIT-P26: drag against the original pixels only. Cropping the source is a
        # slice plus a nearest resize, so the crop follows the pointer exactly and no
        # per-motion basic/mixer/balance work runs. The right half shows the same
        # crop with a progress marker until release triggers the exact comparison.
        if self._pan_source is not None and self._quick_pan():return
        self.changed()

    def release(self,event=None):
        was=self.drag_origin is not None
        self.drag_origin=None
        if not was:return
        # Keep the arc running until the exact comparison arrives, so the right
        # half visibly means "still working" instead of silently showing stale pixels.
        self._spin_pending=True;self._spin_deadline=time.monotonic()+120.
        self._pan_active=False
        self.changed()

    def _quick_pan(self):
        # Draw the source crop immediately for both halves.
        try:
            source=self._pan_source
            width,height,scale,size=self.geometry_values()
            bx,by,bw,bh=crop_bounds(source.shape,self.center,size)
            crop=source[by:by+bh,bx:bx+bw]
            values=((crop.astype(np.uint32)+128)//257).astype(np.uint8) if crop.dtype==np.uint16 else crop
            dest=(max(1,round(bw*scale)),max(1,round(bh*scale)))
            frame=Image.fromarray(values).resize(dest,Image.Resampling.NEAREST)
            out=Image.new('RGB',(width,height),'#151517')
            top=(height-dest[1])//2;pane=width//2
            if self._pan_processed_only:
                out.paste(frame,((width-dest[0])//2,top))
            else:
                out.paste(frame,((pane-dest[0])//2,top))
                out.paste(frame,(pane+1+(width-pane-1-dest[0])//2,top))
            self._pan_active=True
            self._pan_painting=True
            try:self.paint(out,processed_only=self._pan_processed_only)
            finally:self._pan_painting=False
            if not self._pan_processed_only:self._draw_spin()
            return True
        except Exception:
            return False

    def _stop_spin(self):
        if self._spin_after is not None:
            try:self.after_cancel(self._spin_after)
            except tk.TclError:pass
            self._spin_after=None
        try:self.canvas.delete('spin')
        except tk.TclError:pass

    def _draw_spin(self):
        # Placeholder arc over the right half while the exact crop is pending.
        if not (self._pan_active or self._spin_pending) or self.closed:return
        if self._spin_pending and not self._pan_active and time.monotonic()>self._spin_deadline:
            self._spin_pending=False;self._stop_spin();return
        try:
            width=self.canvas.winfo_width();height=self.canvas.winfo_height()
        except tk.TclError:
            return
        if width<20 or height<20:return
        try:self.canvas.delete('spin')
        except tk.TclError:return
        cx=width*3//4;cy=height//2;r=14
        self.canvas.create_arc(cx-r,cy-r,cx+r,cy+r,start=self._spin_angle,extent=260,
            style='arc',outline='#ffffff',width=3,tags='spin')
        self.canvas.create_text(cx,cy+26,text='正在降噪',fill='#d0d0d5',tags='spin')
        self._spin_angle=(self._spin_angle+28)%360
        try:self._spin_after=self.after(90,self._draw_spin)
        except tk.TclError:self._spin_after=None

    def placeholder(self,text):
        self.stop_display()
        self.canvas.delete('all');self.bitmap=None
        w,h,_,_=self.geometry_values()
        self.canvas.create_text(w/2,h/2,text=text,fill='#aeaeb2',width=max(80,w-16),justify='center')

    def stop(self):
        if self.cancel_event:self.cancel_event.set()

    def stop_display(self):
        if self.display_cancel:self.display_cancel.set()
        self.display_request=None;self.display_target=None;self.display_failed=None
        self.display_error=''

    def display_idle(self):
        return self.display_future is None and self.display_request is None and (
            self.display_target is None or self.last_draw==self.display_target or self.display_failed==self.display_target)

    def queue_display(self,key,source,result,amount,alpha,bounds,settings,balance,geometry,processed_only=False):
        # AUDIT-P26: keep the source so a drag can crop it without recomputing anything.
        self._pan_source=source;self._pan_processed_only=bool(processed_only)
        if key!=self.display_target:
            if self.display_cancel:self.display_cancel.set()
            self.display_target=key;self.display_failed=None
            # One replaceable request; a drag does not append work to a queue.
            self.display_request=(key,source,result,amount,alpha,bounds,settings,balance,geometry,processed_only)
        self.poll_display()

    def poll_display(self):
        if self.display_future is not None and self.display_future.done():
            future=self.display_future;key=self.display_running
            mode=self.display_running_mode;serial=self.display_running_serial
            self.display_future=None;self.display_running=None;self.display_cancel=None
            self.display_running_mode=None;self.display_running_serial=None
            try:
                frame=future.result()
                if not self.closed and key==self.display_target and mode==self.source_mode and serial==self._mode_serial:
                    self.paint(frame,processed_only=key[-1]);self.last_draw=key
            except RenderCancelled:pass
            except Exception as exc:
                if not self.closed and key==self.display_target and mode==self.source_mode and serial==self._mode_serial:
                    self.display_failed=key;self.display_error='局部调整预览失败：'+str(exc);self.note.set(self.display_error)
        if not self.closed and self.display_future is None and self.display_request is not None:
            key,source,result,amount,alpha,bounds,settings,balance,geometry,processed_only=self.display_request;self.display_request=None
            event=threading.Event();self.display_cancel=event;self.display_running=key
            self.display_running_mode=self.source_mode;self.display_running_serial=self._mode_serial
            # Only arrays, primitive settings and an Event cross this boundary.
            # A worker finishing after close cannot own the last Tk reference.
            self.display_future=self.display_pool.submit(render_display,source,result,amount,alpha,bounds,
                settings,balance,geometry,event.is_set,processed_only)

    def native_custom(self):
        a=self.app
        if getattr(a,'editor_enabled',False):
            state=a.editor_state
            return (state['denoise']['kind']=='native' and state['filter']['brand']==PENTAX
                    and state['filter']['look'] in LOOKS)
        # Constructor/test clients may not expose the full workbench contract.
        if not hasattr(a,'brand') or not hasattr(a,'look') or not hasattr(a,'overrides'):return False
        return (a.brand.get()==PENTAX and a.look.get() in LOOKS and
                bool(a.native_noise_values()))

    def poll(self):
        if self.closed or not self.app.ready:return
        # Resolve the selected method before consuming asynchronous results.
        # Historical-state changes need not call set_source_mode directly.
        if getattr(self.app,'editor_enabled',False):
            kind=self.app.editor_state['denoise']['kind']
            if kind in ('native','ai'):
                wanted='native' if kind=='native' else 'input'
                if self.source_mode!=wanted:self.set_source_mode(wanted)
        if self.future is not None and self.future.done():
            future=self.future;key=self.running;kind=self.running_kind
            self.future=None;self.running=None;self.cancel_event=None;self.running_kind=None
            try:
                result=future.result()
                if key==self.target:
                    if kind=='native':self.native_reference=result;self.native_reference_key=key
                    else:self.input_preview=result;self.input_preview_key=key
                    self.failed_key=None;self.failed_message='';self.last_draw=None
            except RenderCancelled:pass
            except Exception as exc:
                if key==self.target:
                    self.failed_key=key;self.failed_message='局部预览失败：'+str(exc)
                    self.note.set(self.failed_message)
        while not self.events.empty():
            key,text=self.events.get()
            if key==self.target and self.future is not None:self.note.set(text)
        if not self.app.current_path:
            self.needs_detail=False;self.stop()
            if self._last_note!='empty':self.placeholder('等待照片');self._last_note='empty'
            return
        if getattr(self.app,'editor_enabled',False):
            kind=self.app.editor_state['denoise']['kind']
            if kind=='off':
                self.needs_detail=False;self.stop();self.target=None
                self.native_reference=None;self.native_reference_key=None
                self.heading.set('细节预览')
                if self._last_note!='denoise-off':
                    self.placeholder('降噪已关闭');self.note.set('');self._last_note='denoise-off'
                return
            wanted='native' if kind=='native' else 'input'
            if self.source_mode!=wanted:self.set_source_mode(wanted)
        if self.source_mode=='input':self.poll_input()
        else:self.poll_native()

    def poll_input(self):
        a=self.app;self.needs_detail=False
        history_only=bool(getattr(a,'editor_enabled',False) and getattr(a,'editor_history_only',False))
        settings=a.current_photo_denoise();store=a.photo_denoise_store;key=a.denoise_key()
        if key!=self._input_context_key:
            self.stop();self.target=None;self.failed_key=None;self.failed_message=''
            self._input_context_key=key;self._input_task_id=None;self._mode_serial+=1
            self.placeholder('正在准备输入降噪对比…');self.note.set('');self._last_note='input-context'
        if key is None:
            self.stop();self.target=None
            if self._last_note!='input-off':
                self.placeholder('未应用输入降噪');self.note.set('');self._last_note='input-off'
            return
        status=store.status(key);state=status['state'];task_id=status.get('task_id')
        active=state in ('queued','preparing','running')
        if task_id!=self._input_task_id:
            self._input_task_id=task_id
            if active:self.failed_key=None;self.failed_message=''
        if state in ('failed','cancelled'):
            # An unsuccessful full job is not permission to silently start a
            # second inference on a crop. A previous local result is deliberately
            # hidden: it must not be presented as this failed run's final result.
            self.stop();self.target=('input-terminal',key,task_id,state)
            label='输入降噪失败' if state=='failed' else '输入降噪已取消'
            reason=str(status.get('error') or '').strip()
            message=label+('：'+reason if reason else '')+'；点击“开始全尺寸降噪”重试。'
            marker=('input-terminal',key,task_id,state,message)
            if self._last_note!=marker:
                self.input_preview=None;self.input_preview_key=None
                self.placeholder(label+'\n可重新开始全尺寸降噪');self._last_note=marker
            self.note.set(message);return
        held=a.current_denoise()
        record=held['record'] if held is not None else None
        if (record is None and not active and self.failed_key is not None
                and self.failed_key[:2]==('input',key)):
            # Preserve the failure even if the prepared-source cache is later
            # evicted or a Configure/drag event changes the requested crop.
            # A new explicit full-job attempt or a different model clears it.
            marker=('input-local-failed',self.failed_key)
            if self._last_note!=marker:
                self.placeholder('局部预览失败\n可重新开始全尺寸降噪');self._last_note=marker
            self.note.set(self.failed_message+'；点击“开始全尺寸降噪”重试。');return
        prepared=record.prepared if record is not None else store.get_source(key.identity)
        if prepared is None:
            if history_only:
                self.note.set('此步骤的降噪结果未缓存，需重新计算')
                if self._last_note!='history-input-missing':
                    self.placeholder('此步骤的降噪结果未缓存，需重新计算');self._last_note='history-input-missing'
                return
            store.request_prepare(a.current_path)
            self.note.set('正在准备输入照片…');return
        source=prepared.rgb;alpha=prepared.alpha
        self._size=(source.shape[1],source.shape[0]);geometry=self.geometry_values()
        bounds=crop_bounds(source.shape,self.center,geometry[3]);target=('input',key,self._input_revision,bounds)
        if target!=self.target:
            self.stop();self.target=target;self.last_draw=None
            if self.failed_key!=target:self.failed_key=None;self.failed_message=''
        result=None;result_stamp=None
        if record is not None:
            result={'source':source,'clean':record.rgb,'bounds':(0,0,source.shape[1],source.shape[0]),
                    'context_source':source,'context_clean':record.rgb,
                    'context_bounds':(0,0,source.shape[1],source.shape[0]),'full':True}
            result_stamp=id(record)
        elif self.input_preview_key==target:
            result=self.input_preview;result_stamp=id(result['clean'])
        elif self.input_preview_key is not None and self.input_preview_key[:3]==target[:3]:
            result=crop_result(self.input_preview,bounds,source.shape)
            if result is not None:result_stamp=id(self.input_preview['context_clean'])
        if result is None and hasattr(store,'get_preview'):
            candidate=store.get_preview(key)
            if candidate is not None:
                result=crop_result(candidate,bounds,source.shape);result_stamp=id(candidate['clean'])
        if result is None and history_only:
            self.note.set('此步骤的降噪结果未缓存，需重新计算')
            if self._last_note!='history-input-missing':
                self.placeholder('此步骤的降噪结果未缓存，需重新计算');self._last_note='history-input-missing'
            return
        if result is None and not active and settings['denoise'] and self.future is None and self.failed_key!=target:
            event=threading.Event();self.cancel_event=event;self.running=target;self.running_kind='input'
            center=self.center;size=geometry[3];model=settings['denoise_model'];events=self.events
            def progress(text):events.put((target,str(text)))
            self.future=self.pool.submit(render_patch,source,center,size,model,event.is_set,progress)
        if result is None:
            self.note.set('全片处理中…' if active else '正在试算局部细节…')
            if self.bitmap is None and self._last_note!='input-waiting':
                self.placeholder('正在准备降噪对比…');self._last_note='input-waiting'
            return
        draw_key=(target,settings['denoise'],result_stamp,geometry[:3],False)
        self.queue_display(draw_key,source,result,settings['denoise'],alpha,bounds,{},(0,0),geometry[:3])
        self.note.set(self.display_error if self.display_failed==draw_key else '已应用' if record is not None else '局部预览')
        self._last_note='input-image'

    def poll_native(self):
        a=self.app
        if getattr(a,'editor_enabled',False):
            return self.poll_native_editor()
        if a.brand.get()!=PENTAX or a.look.get() not in LOOKS:
            self.needs_detail=False;self.note.set('当前滤镜没有原厂降噪对照');return
        cache=a.detail_cache
        valid=cache is not None and cache[:2]==(a.generation,a.requested) and a.signature()==a.requested
        self.needs_detail=not valid and not a.preview_error
        if not valid:
            self.note.set('正在更新原厂结果…');return
        key,ev,items,amount=a.requested
        colour={k:v for k,v in items if k not in NR_CONTROLS}
        snapshot=a.source_snapshot()
        reference_key=('native',a.source_token(snapshot),key,ev,tuple(sorted(colour.items())),amount)
        if self.target!=reference_key:
            self.stop();self.target=reference_key;self.failed_key=None;self.last_draw=None
        if not any(k in NR_CONTROLS for k,v in items):
            reference=cache[3]
        elif self.native_reference_key==reference_key:
            reference=self.native_reference
        else:
            reference=None
            if self.future is None and self.failed_key!=reference_key:
                event=threading.Event();self.cancel_event=event;self.running=reference_key;self.running_kind='native'
                self.future=self.pool.submit(native_reference,snapshot,key,ev,colour,amount,event.is_set)
        if reference is None:
            if self.failed_key!=reference_key:self.note.set('正在准备同设置对照…')
            return
        source=reference;clean=cache[3]
        if source.shape!=clean.shape:self.note.set('对照尺寸不一致');return
        self._size=(source.shape[1],source.shape[0]);geometry=self.geometry_values()
        bounds=crop_bounds(source.shape,self.center,geometry[3]);full_bounds=(0,0,source.shape[1],source.shape[0])
        result={'source':source,'clean':clean,'bounds':full_bounds,'context_source':source,
                'context_clean':clean,'context_bounds':full_bounds,'full':True}
        settings=a.render_post_settings();balance=tuple(a.current_balance())
        draw_key=(reference_key,id(cache),bounds,repr(settings),balance,geometry[:3],False)
        self.queue_display(draw_key,source,result,100,cache[4],bounds,settings,balance,geometry[:3])
        self.note.set(self.display_error if self.display_failed==draw_key else '');self._last_note='native-image'

    def poll_native_editor(self):
        a=self.app;state=deepcopy(a.editor_state)
        if state['denoise']['kind']!='native':
            self.needs_detail=False;return
        if state['filter']['brand']!=PENTAX or state['filter']['look'] not in LOOKS:
            self.needs_detail=False;self.stop();self.target=None
            self.note.set('当前滤镜没有原厂降噪对照');return
        cache=a.detail_cache
        valid=cache is not None and cache[:2]==(a.generation,a.requested) and a.signature()==a.requested
        self.needs_detail=not valid and not a.preview_error
        if not valid:
            self.note.set('正在更新原厂结果…');return
        from photo_edit import fingerprint
        try:reference_state=editor_reference_state(state)
        except ValueError as exc:
            self.needs_detail=False;self.note.set(str(exc));return
        # The default-NR comparison does not depend on the moving custom NR
        # sliders. Reuse it while manual edits, the look and this photo stay put.
        reference_key=('native-editor',a.generation,fingerprint(reference_state))
        if self.target!=reference_key:
            self.stop();self.target=reference_key;self.failed_key=None;self.last_draw=None
        if not state['denoise']['native']:
            reference=cache[3]
        elif self.native_reference_key==reference_key:
            reference=self.native_reference
        else:
            reference=None
            if self.future is None and self.failed_key!=reference_key:
                event=threading.Event();self.cancel_event=event;self.running=reference_key;self.running_kind='native'
                path=a.current_path;store=a.photo_denoise_store;library=a.custom_library;events=self.events
                def progress(text):events.put((reference_key,str(text)))
                self.future=self.pool.submit(editor_native_reference,path,store,library,state,event.is_set,progress)
        if reference is None:
            if self.failed_key!=reference_key:self.note.set('正在准备同设置对照…')
            return
        source=reference;clean=cache[3]
        if source.shape!=clean.shape or source.dtype!=clean.dtype:
            self.note.set('对照的尺寸或精度不一致');return
        self._size=(source.shape[1],source.shape[0]);geometry=self.geometry_values()
        bounds=crop_bounds(source.shape,self.center,geometry[3]);full_bounds=(0,0,source.shape[1],source.shape[0])
        result={'source':source,'clean':clean,'bounds':full_bounds,'context_source':source,
                'context_clean':clean,'context_bounds':full_bounds,'full':True}
        # EditorSession applied all manual adjustments before the filter.
        # Applying them again here would fabricate the denoise comparison.
        settings=validate_post();balance=(0,0)
        draw_key=(reference_key,id(cache),bounds,geometry[:3],False)
        self.queue_display(draw_key,source,result,100,cache[4],bounds,settings,balance,geometry[:3])
        self.note.set(self.display_error if self.display_failed==draw_key else '');self._last_note='native-image'


    def paint(self,out,processed_only=False):
        # A paint outside _quick_pan is a real result: the pending arc can stop.
        if not getattr(self,'_pan_painting',False) and getattr(self,'_spin_pending',False):
            self._spin_pending=False;self._stop_spin()
        width,height=out.size
        self.display_processed_only=bool(processed_only)
        self.heading.set('输入降噪' if self.source_mode=='input' else '原厂降噪')
        self.bitmap=ImageTk.PhotoImage(out,master=self.canvas);self.canvas.delete('all')
        self.canvas.create_image(width/2,height/2,image=self.bitmap)
        if processed_only:
            self.canvas.create_text(4,4,text='原厂处理后',anchor='nw',fill='#ffffff')
            return
        self.canvas.create_line(width/2,0,width/2,height,fill='#ffffff')
        captions=('原厂默认','当前设置') if self.source_mode=='native' else ('降噪前','降噪后')
        for cx,label in ((4,captions[0]),(width-4,captions[1])):
            self.canvas.create_text(cx,4,text=label,anchor='nw' if cx==4 else 'ne',fill='#ffffff')

    def close(self):
        self.closed=True;self.needs_detail=False;self.stop();self.stop_display()
        self.pool.shutdown(wait=False,cancel_futures=True)
        self.display_pool.shutdown(wait=False,cancel_futures=True)
