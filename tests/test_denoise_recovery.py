"""Photo-denoise replacement and speculative startup work stay independent."""
import hashlib
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from editor_ui import EditorMixin
from photo_denoise import PhotoDenoiseStore, _write_tiff, key_for
from photo_edit import default_state
from source_denoise_ui import SourceDenoiseMixin


class DenoiseRecoveryTests(unittest.TestCase):
    def test_main_preview_keeps_same_input_interim_but_rejects_old_denoise_input(self):
        from workbench_app import App
        class Harness:
            poll_once=App.poll_once
            def __init__(self):
                self.ready=False;self.closed=False;self.generation=1;self.editor_enabled=True
                self.requested=('Natural',0.,(),1.,'current');self.native_cache={};self.factory={}
                self.events=queue.Queue();self.preview_times={};self.photo_started=time.monotonic()
                self.current_path=None;self.preview_started=1;self.busy=False;self.custom_importing=False
                self.after='current-frame';self.calls=[]
            def source_token(self):return ('photo','current-input',100.)
            def cache_result(self,*args):self.calls.append(('cache',args[1]))
            def queue_post(self,image):self.after=image;self.calls.append(('display',image))
            def draw(self):pass
            def denoise_idle(self):return True
        app=Harness();old=('Natural',0.1,(),1.,'old-parameter')
        app.events.put(('native_result',(1,old,('before','wrong-input',{'_editor_input_token':('photo','old-input',100.)}))))
        app.poll_once()
        self.assertEqual(app.after,'current-frame')
        self.assertFalse(app.calls)
        app.events.put(('native_result',(1,old,('before','same-input',{'_editor_input_token':app.source_token()}))))
        app.poll_once()
        self.assertEqual(app.after,'same-input')
        self.assertIn(('display','same-input'),app.calls)

    def test_new_editor_parameters_emit_fast_frame_before_detail_frame(self):
        state=default_state();sig=EditorMixin._signature_from(state,None);calls=[]
        class Session:
            decode_count=1
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def render(self,state,maxsize=None,**kwargs):
                calls.append(maxsize)
                pixels=np.zeros((4,4,3),np.uint16)
                return pixels,pixels,None,{}
        class Harness(EditorMixin):
            def needs_full_detail(self):return False
            def source_token(self,snapshot):return SourceDenoiseMixin.source_token(self,snapshot)
        app=Harness();app.closed=False;app.busy=False;app.generation=1
        app.current_path='photo.tif';app.cache_evictions=queue.SimpleQueue();app.detail_cache=None
        app.photo_denoise_store=None;app.custom_library=None
        app._editor_interacting_until=0.;app._editor_requested=(sig,state,None)
        app.requested=sig;app.request_revision=1;app.events=queue.Queue()
        with patch('editor_pipeline.EditorSession',Session), \
             patch('native_app.preview',return_value=Image.new('RGB',(4,4))), \
             patch('tone_tools.histogram',return_value={}):
            worker=threading.Thread(target=app.load_editor_worker,args=('photo.tif',1))
            worker.start()
            deadline=time.monotonic()+2
            while app.events.qsize()<4 and time.monotonic()<deadline:time.sleep(.005)
            app.closed=True;worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(calls[:2],[(480,360),(960,720)],list(app.events.queue))
        levels=[value[2][2]['_editor_display_level'] for kind,value in list(app.events.queue)
                if kind=='native_result']
        self.assertEqual(levels[:2],[1,2])

    def test_discard_and_regenerate_same_model_replaces_pixels_and_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            photo=Path(folder)/'input.tif'
            rgb=np.arange(32*48*3,dtype=np.uint16).reshape(32,48,3)
            _write_tiff(photo,rgb,None)
            original=hashlib.sha256(photo.read_bytes()).digest()
            calls=[]

            def infer(source,model,mode,cancelled=None,progress=None,on_status=None):
                calls.append((model,mode))
                return np.minimum(source.astype(np.uint32)+len(calls)*37,65535).astype(np.uint16),{'backend':'test'}

            with patch('neural_process.infer',side_effect=infer):
                store=PhotoDenoiseStore()
                try:
                    first=store.get_or_compute(photo,'nafnet','auto')
                    key=key_for(photo,'nafnet','auto')
                    first_token=SourceDenoiseMixin.source_token(None,(photo,first,100))
                    self.assertTrue(store.discard(key))
                    self.assertIsNone(store.get(key))
                    self.assertIsNone(store.get_preview(key))
                    self.assertEqual(store.status(key)['state'],'missing')
                    self.assertTrue(first.path.is_file())  # An existing reader still owns its file.
                    second=store.get_or_compute(photo,'nafnet','auto')
                    self.assertEqual(len(calls),2)
                    self.assertNotEqual(first.run_id,second.run_id)
                    self.assertNotEqual(first_token,SourceDenoiseMixin.source_token(None,(photo,second,100)))
                    self.assertFalse(np.array_equal(first.rgb,second.rgb))
                    self.assertEqual(hashlib.sha256(photo.read_bytes()).digest(),original)
                finally:store.close()

    def test_editor_signature_uses_result_run_not_only_photo_model(self):
        state=default_state();state['denoise'].update(kind='ai',model='nafnet',amount=100.)
        record=SimpleNamespace(key='same-photo-and-model',run_id='first')
        holder=SimpleNamespace(editor_state=state,current_denoise=lambda:{'key':record.key,'record':record})
        first=EditorMixin.editor_signature(holder)
        record.run_id='second'
        self.assertNotEqual(first,EditorMixin.editor_signature(holder))

    def test_ui_discard_clears_lease_and_records_off_state(self):
        with tempfile.TemporaryDirectory() as folder:
            photo=Path(folder)/'input.tif'
            _write_tiff(photo,np.zeros((8,8,3),np.uint16),None)
            key=key_for(photo,'nafnet','auto')
            store=PhotoDenoiseStore()
            try:
                store._results[key]=SimpleNamespace(key=key,source=None,rgb=None,alpha=None)
                class Harness(SourceDenoiseMixin):
                    def refresh_source_render(self):self.renders+=1
                    def refresh_denoise_controls(self):pass
                    def editor_record(self,state,label):
                        self.editor_state=state;self.recorded=label;return True
                app=Harness();app.current_path=str(photo);app.busy=False;app.editor_enabled=True
                app.editor_state=default_state();app.editor_state['denoise'].update(kind='ai',model='nafnet')
                app.photo_denoise_store=store;app.denoise_result={'key':key,'record':store._results[key]}
                app.source_denoise_keys={key};app._source_completed_keys={key};app._source_status_stamp='old'
                detail=SimpleNamespace(cleared=0)
                detail.clear_input_cache=lambda:setattr(detail,'cleared',detail.cleared+1)
                app.denoise_detail=detail;app.renders=0
                app.denoise_state=SimpleNamespace(set=lambda value:None)
                app.discard_denoise()
                self.assertEqual(app.editor_state['denoise']['kind'],'off')
                self.assertEqual(app.recorded,'丢弃降噪结果')
                self.assertIsNone(app.denoise_result)
                self.assertIsNone(store.get(key))
                self.assertEqual(detail.cleared,1)
                self.assertEqual(app.renders,1)
            finally:store.close()

    def test_regenerate_replaces_cached_result_and_failure_can_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            photo=Path(folder)/'input.tif'
            _write_tiff(photo,np.zeros((16,16,3),np.uint16),None)
            calls=[]
            def infer(source,model,mode,cancelled=None,progress=None,on_status=None):
                calls.append(len(calls)+1)
                if len(calls)==3:raise RuntimeError('inference failed')
                return np.full_like(source,len(calls)*111),{'backend':'test'}
            with patch('neural_process.infer',side_effect=infer):
                store=PhotoDenoiseStore()
                try:
                    first=store.get_or_compute(photo,'nafnet','auto')
                    class Harness(SourceDenoiseMixin):
                        def refresh_source_render(self):self.renders+=1
                        def start_denoise(self):
                            self.started+=1
                            self.photo_denoise_store.request(self.current_path,'nafnet','auto')
                    app=Harness();app.current_path=str(photo);app.busy=False;app.editor_enabled=True
                    app.editor_state=default_state();app.editor_state['denoise'].update(kind='ai',model='nafnet')
                    app.photo_denoise_store=store;app.denoise_result={'key':first.key,'record':first}
                    app.source_denoise_keys={first.key};app._source_completed_keys=set()
                    app._source_status_stamp='old';app.denoise_detail=None
                    app.denoise_state=SimpleNamespace(set=lambda value:None);app.renders=0;app.started=0
                    app.regenerate_denoise()
                    until=time.monotonic()+5
                    while store.get(first.key) is None and time.monotonic()<until:time.sleep(.005)
                    second=store.get(first.key)
                    self.assertIsNotNone(second)
                    self.assertNotEqual(first.run_id,second.run_id)
                    self.assertEqual(app.started,1)
                    self.assertIsNone(app.denoise_result)
                    store.discard(first.key)
                    store.request(photo,'nafnet','auto')
                    while store.status(first.key)['state'] in ('queued','preparing','running') and time.monotonic()<until:
                        time.sleep(.005)
                    self.assertEqual(store.status(first.key)['state'],'failed')
                    store.discard(first.key)
                    third=store.get_or_compute(photo,'nafnet','auto')
                    self.assertEqual(calls,[1,2,3,4])
                    self.assertEqual(int(third.rgb[0,0,0]),444)
                finally:store.close()

    def test_discard_active_job_cannot_publish_a_late_old_result(self):
        with tempfile.TemporaryDirectory() as folder:
            photo=Path(folder)/'input.tif'
            _write_tiff(photo,np.zeros((8,8,3),np.uint16),None)
            entered=threading.Event();release=threading.Event();calls=[]
            from native_backend import RenderCancelled
            def infer(source,model,mode,cancelled=None,progress=None,on_status=None):
                calls.append(len(calls)+1);entered.set()
                while not release.wait(.005):
                    if cancelled():raise RenderCancelled()
                return np.full_like(source,calls[-1]),{'backend':'test'}
            with patch('neural_process.infer',side_effect=infer):
                store=PhotoDenoiseStore()
                try:
                    key=store.request(photo,'nafnet','auto')
                    self.assertTrue(entered.wait(5))
                    store.discard(key)
                    store.request(photo,'nafnet','auto')
                    release.set()
                    until=time.monotonic()+5
                    while store.get(key) is None and time.monotonic()<until:time.sleep(.005)
                    self.assertEqual(calls,[1,2])
                    self.assertEqual(int(store.get(key).rgb[0,0,0]),2)
                finally:store.close()

    def test_background_start_waits_for_colour_probe_and_foreground_idle(self):
        from workbench_app import App
        class Harness:
            start_background_preparation=App.start_background_preparation
            _schedule_background_preparation=lambda self,delay=400:self.scheduled.append(delay)
            post_idle=lambda self:True
            denoise_idle=lambda self:True
        app=Harness();app.closed=False;app.busy=False;app.preview_started=0
        app._background_preparation_after=None;app._background_idle_at=0.
        app._color_warmup_thread=None;app._prewarm_attempted=False;app.scheduled=[]
        running=SimpleNamespace(is_alive=lambda:True)
        with patch('gpu_compute.status',return_value={'mode':'auto','state':'waiting'}), \
             patch('gpu_compute.start_background_warmup',return_value=running) as color, \
             patch('startup_prewarm.start',return_value=True) as ai:
            app.start_background_preparation()
            color.assert_called_once();ai.assert_not_called()
        app._color_warmup_thread=SimpleNamespace(is_alive=lambda:False)
        with patch('gpu_compute.status',return_value={'mode':'auto','state':'ready'}), \
             patch('startup_prewarm.start',return_value=True) as ai:
            app.start_background_preparation()
            ai.assert_called_once_with(delay=0.)
            self.assertTrue(app._prewarm_attempted)
        app._prewarm_attempted=False;app._background_idle_at=time.monotonic()+2
        with patch('startup_prewarm.start') as ai:
            app.start_background_preparation()
            ai.assert_not_called()


if __name__=='__main__':unittest.main()
