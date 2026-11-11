"""Bounded preview scheduling, independent of original-size export processing."""

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

import json
import sys
import time
from PIL import Image
import numpy as np
import gpu_compute
from native_backend import RenderCancelled
from postprocess import preview as post_preview
from tone_tools import histogram
from filter_catalog import PENTAX


class PreviewAdjustmentsMixin:
    def schedule_post(self,delay=45):
        if hasattr(self,'defer_background_preparation'):self.defer_background_preparation()
        # Throttle continuous changes; resetting a trailing timer on every
        # slider event would prevent any image update until the user stopped.
        self._post_interacting_until=time.monotonic()+.16
        if self.balance_pending is None:
            self.balance_pending=self.root.after(delay,self.refresh_balance)

    def queue_post(self,image):
        if getattr(self,'_post_input',None) is not image:
            context=(self.generation,self.requested)
            if getattr(self,'_post_source_context',None)!=context:self.after=None
            self._post_source_context=context;self._post_input=image;self._post_stage_cache={}
            self._post_interim_done=None
        self.post_revision+=1;serial=self.post_revision
        settings=self.render_post_settings() if hasattr(self,'render_post_settings') else dict(self.current_post(),denoise=0)
        settings=dict(settings,denoise=0);clean=None
        request=(serial,self.generation,self.requested,image,self.current_balance(),settings,clean)
        self.post_request=request
        if self.post_future is None:self.submit_post()
        # Original-size colour inspection starts now, not after the fit frame.
        viewport=getattr(self,'viewport',None)
        if viewport is not None and viewport.scale:
            # Queue work without recreating a large Tk bitmap on every value.
            # The completed frame, or a geometry gesture, schedules painting.
            refresh=getattr(viewport,'parameters_changed',None)
            if callable(refresh):refresh()
            else:self.draw()

    def post_idle(self):
        return self.balance_pending is None and self.post_future is None and self.post_request is None and self.post_completed==self.post_revision

    def submit_post(self):
        request=self.post_request
        if request is None or self.post_future is not None:return
        serial,generation,sig,image,balance,settings,clean=request
        interacting=time.monotonic()<getattr(self,'_post_interacting_until',0.)
        # One temporary frame per revision. Keep its exact final request queued
        # until the interaction pauses, rather than recomputing the same frame.
        if interacting and getattr(self,'_post_interim_done',None)==serial:return
        self.post_request=None
        self.post_running=request
        limit=getattr(self,'_post_interactive_limit',480)
        ratio=min(1.,limit/max(image.size)) if interacting else 1.
        target=(max(1,round(image.width*ratio)),max(1,round(image.height*ratio)))
        interim=target!=image.size
        # Source-pixel scale controls spatial radii. Zoom magnification does
        # not belong here; the 100% viewport has its own original-pixel path.
        detail=self.detail_cache
        if detail is not None and detail[:2]==(generation,sig):
            native_size=(detail[2].shape[1],detail[2].shape[0])
        else:
            report=self.factory.get(sig[0],{})
            native_size=(report.get('native_width',image.width),report.get('native_height',image.height))
        scale=min(target[0]/max(1,native_size[0]),target[1]/max(1,native_size[1]))
        cache=getattr(self,'_post_stage_cache',None)
        if cache is None:cache={};self._post_stage_cache=cache
        context=(generation,sig,id(image),target,scale)
        def cancelled():
            # A newer slider value from this same source may finish after this
            # frame. Cancelling on every serial change starves slow machines.
            return (self.closed or generation!=self.generation or sig!=self.requested
                    or getattr(self,'_post_input',None) is not image)
        def run():
            started=time.perf_counter()
            if cancelled():raise RenderCancelled()
            # The executor is single-threaded. Retain only the latest result
            # of each stage, bounded to four fit-sized images. Unchanged early
            # stages can be reused when the user changes later-stage controls.
            def cached(name,key,render):
                previous=cache.get(name)
                if previous is not None and previous[0]==key:return previous[1]
                result=render()
                if cancelled():raise RenderCancelled()
                cache[name]=(key,result)
                return result
            base_key=(context,)
            def base_image():
                # Input already includes source denoise and the chosen filter.
                # A preview must never substitute a raw clean image over it.
                return image.resize(target,Image.Resampling.BILINEAR) if interim else image
            source=cached('base',base_key,base_image)
            # These calls preserve the production operation order and each
            # stage's quantization. Never run a denoiser on a fit preview.
            basic_key=(base_key,tuple(sorted(settings['basic'].items())))
            basic=cached('basic',basic_key,lambda:post_preview(source,(0,0),{'basic':settings['basic']},cancelled,scale=scale))
            mixer_key=(basic_key,json.dumps(settings['mixer'],sort_keys=True,separators=(',',':')))
            mixed=cached('mixer',mixer_key,lambda:post_preview(basic,(0,0),{'mixer':settings['mixer']},cancelled,scale=scale))
            balance_key=(mixer_key,tuple(balance))
            balanced=cached('balance',balance_key,lambda:post_preview(mixed,balance,{},cancelled,scale=scale))
            after=post_preview(balanced,(0,0),{'tone':settings['tone']},cancelled,scale=scale)
            if cancelled():raise RenderCancelled()
            # Keep display geometry steady while using a temporary interactive
            # resolution. It is never marked completed or used for export.
            if interim:after=after.resize(image.size,Image.Resampling.BILINEAR)
            hist=histogram(after)
            return serial,generation,sig,after,hist,interim,time.perf_counter()-started
        self.post_future=self.post_pool.submit(run)

    def poll_post(self):
        future=self.post_future
        if future is not None and future.done():
            running=self.post_running;self.post_future=None;self.post_running=None
            try:
                serial,generation,sig,after,hist,*extra=future.result()
                interim=bool(extra and extra[0])
                current=(generation==self.generation and sig==self.requested and running
                         and running[3] is getattr(self,'_post_input',None))
                if current:
                    if interim:
                        self._post_interim_done=serial
                        elapsed=extra[1] if len(extra)>1 else 0.
                        limit=getattr(self,'_post_interactive_limit',480)
                        if elapsed>.18:self._post_interactive_limit=max(320,int(limit*.75)//16*16)
                        elif elapsed and elapsed<.07:self._post_interactive_limit=min(640,int(limit*1.25)//16*16)
                    else:self.post_completed=max(self.post_completed,serial)
                    if serial>=getattr(self,'_post_presented',-1) and not self.preview_error:
                        self._post_presented=serial
                        self.after=after;self.output_histogram=(generation,after,hist);self.draw()
                    if interim and serial==self.post_revision and self.post_request is None:
                        self.post_request=running
            except RenderCancelled:pass
            except Exception as exc:
                if (running and running[:3]==(self.post_revision,self.generation,self.requested)
                        and running[3] is getattr(self,'_post_input',None)):
                    self.post_completed=self.post_revision;self.preview_error=str(exc);self.status.set('后置调整失败：'+str(exc));self.after=None;self.draw()
        if self.post_future is None and self.post_request is not None:self.submit_post()
        state=gpu_compute.status()
        text='CPU' if state['mode']=='cpu' else ('GPU  '+state['device'] if state['state']=='ready' else 'CPU  '+state['reason'])
        if state.get('external_busy'):text='GPU 正在降噪；调色暂用 CPU 保持响应'
        if self.brand.get()==PENTAX:text+='\n宾得引擎  CPU'
        # The background worker imports this module for the first time. Python
        # publishes it in sys.modules before its status function is defined.
        # Observing that partial module must not terminate the UI polling loop.
        denoise_status=getattr(sys.modules.get('denoise'),'status',None)
        if callable(denoise_status) and self.current_post()['denoise']:
            state_dn=denoise_status()
            if state_dn.get('backend') and state_dn.get('model')==self.current_post()['denoise_model']:text+='\n降噪：'+state_dn['backend']
        if self.acceleration_text.get()!=text:self.acceleration_text.set(text)
