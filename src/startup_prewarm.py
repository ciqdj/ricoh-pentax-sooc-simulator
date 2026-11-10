"""Prepare the preferred AI model in the same persistent inference worker."""
from pathlib import Path
import json,os,threading,time

_guard=threading.RLock()
_stop=threading.Event()
_thread=None
_status={}


def preference():
    from app_paths import CONFIG
    try:
        value=json.loads((CONFIG/'ai-prewarm.json').read_text(encoding='utf-8'))
        if value.get('model') in ('nafnet','restormer'):return value['model']
    except (OSError,ValueError,TypeError,AttributeError):pass
    return 'nafnet'


def snapshot():
    with _guard:
        result=dict(_status)
        if result.get('state')=='running':result['elapsed']=time.monotonic()-result['started']
        return result


def stop(wait=False):
    _stop.set()
    thread=_thread
    if wait and thread is not None and thread is not threading.current_thread():thread.join(timeout=5)


def before_infer(model,mode):
    # A matching real job reuses the ongoing preparation and its live status.
    # A different model takes priority and cancels the speculative job.
    with _guard:
        if _status.get('state')!='running' or (_status.get('model'),_status.get('mode'))!=(model,mode):_stop.set()
    from app_paths import CONFIG
    try:
        target=CONFIG/'ai-prewarm.json';target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps({'model':model}),encoding='utf-8');temporary.replace(target)
    except OSError:pass


def start(model=None,mode='auto',delay=2.):
    global _thread,_stop,_status
    if os.environ.get('PHOTO_PREWARM_DISABLE')=='1':return
    model=model or preference()
    if model not in ('nafnet','restormer') or mode not in ('auto','cpu'):return
    with _guard:
        if _thread is not None and _thread.is_alive():return
        _stop=threading.Event();cancel=_stop
        _status={'state':'queued','model':model,'mode':mode,'name':'模型预热','task_id':'ai-prewarm',
                 'stage':'等待启动完成','elapsed':0.,'eta_seconds':None}
    def run():
        import neural_process as host
        import gpu_compute
        from native_backend import RenderCancelled
        acquired=False;busy=False
        def update(**values):
            with _guard:_status.update(values)
        def progress(text):update(stage=text)
        def status(info):update(backend=info.get('backend'),phase=info.get('phase'),eta_seconds=None)
        try:
            if cancel.wait(delay):raise RenderCancelled()
            while not host._lock.acquire(timeout=.1):
                if cancel.is_set():raise RenderCancelled()
            acquired=True
            if cancel.is_set():raise RenderCancelled()
            update(state='running',started=time.monotonic(),stage='准备 AI 后台运行环境')
            gpu_compute.set_external_busy(True);busy=True
            _,info=host._infer_once(None,model,mode,cancel.is_set,progress,status)
            update(state='completed',stage='模型已就绪',backend=info.get('backend'),
                   seconds_per_tile=info.get('seconds_per_tile'),elapsed=time.monotonic()-_status['started'])
        except RenderCancelled:update(state='cancelled',stage='预热已停止')
        except host.WorkerExited as exc:
            host._record_exit(exc,model)
            if exc.status.get('provider')=='DmlExecutionProvider':host._cpu_after_crash.add(model)
            update(state='failed',stage='模型预热未完成；实际任务将重新准备')
        except Exception as exc:update(state='failed',stage='模型预热未完成',error=type(exc).__name__)
        finally:
            if busy:gpu_compute.set_external_busy(False)
            if acquired:host._lock.release()
    _thread=threading.Thread(target=run,name='AIStartupPreparation',daemon=True);_thread.start()
