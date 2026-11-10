"""Focused R3 regression: source domains, dynamic settings and export precision."""
from pathlib import Path
import json,time
import numpy as np


def main(folder):
    from app_paths import ROOT
    from gr4_renderer import KEYS,point,render,profiles,control_values,tone_count,exposure_table
    from gr4_parameters import calculated,signature
    from gr4_accel import library
    from color_math import srgb_encode
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    with np.load(ROOT/'gr4-data/verification.npz',allow_pickle=False) as fixture:
        for key,profile in KEYS.items():
            for tone in range(tone_count(key)):
                controls={'tone':tone} if tone_count(key)>1 else {}
                curves,matrix=calculated(profile,signature(profile,controls))
                old=profiles()['looks'][profile]
                check(key+str(tone)+' default tables',np.array_equal(curves,old['curves'][tone]) and np.array_equal(matrix,old['saturation']))
                actual=render(fixture['input'],key,controls,allow_gpu=False,input_encoding='standard')
                check(key+str(tone)+' legacy fixture',np.array_equal(actual,fixture[f'{key}__{tone}']))
    source=np.random.default_rng(3511).integers(0,65536,(83,97,3),dtype=np.uint16)
    controls={'key':1,'contrast':-2,'shadow':2,'highlight':-3,'saturation':3}
    for encoding in ('standard','srgb','linear'):
        for ev,amount in ((0,1),(.5,.6),(-1.,1.3),(0,0)):
            actual=render(source,'PositiveFilm',controls,ev,amount,allow_gpu=False,input_encoding=encoding)
            base=source.astype(np.float64)/65535
            if encoding=='linear':base=srgb_encode(base)
            processed=point(exposure_table(ev,encoding)[source],'PositiveFilm',controls=controls,input_encoding=encoding,allow_native=False)
            expected=np.floor(np.clip(base+(processed-base)*amount,0,1)*65535+.5).astype(np.uint16)
            check(f'{encoding} exposure {ev} amount {amount}',np.array_equal(actual,expected))
    x=np.random.default_rng(11).uniform(0,1,(21,27,3))
    check('linear and encoded representations agree',np.max(np.abs(point(x,'Vivid',input_encoding='linear')-point(srgb_encode(x),'Vivid',input_encoding='srgb')))<1e-12)
    for key in KEYS:
        settings=control_values(key);base=point(x,key,allow_native=False)
        for name,value in settings.items():
            candidate=2 if name=='tone' and value!=2 else 0 if name=='tone' else -4 if value!=-4 else 4
            changed=point(x,key,controls={name:candidate},allow_native=False)
            check(key+' '+name+' active',np.max(np.abs(base-changed))>1e-8)
    for key in ('MonotoneGR','SoftMonotone','HardMonotone','HighContrastBW'):
        check(key+' applicable controls','saturation' not in control_values(key))
        check(key+' achromatic',np.ptp(render(source,key,allow_gpu=False).astype(np.int32),axis=-1).max()==0)
    from gpu_compute import warmup,status
    warmup();gpu=status()
    if gpu['state']=='ready':
        from gpu_compute import _backend
        for encoding in ('linear','srgb','standard'):
            actual=_backend.gr4(source,'PositiveFilm',0,.5,.6,controls=controls,input_encoding=encoding)
            expected=render(source,'PositiveFilm',controls,.5,.6,allow_gpu=False,input_encoding=encoding)
            check('GPU '+encoding+' dynamic parity',np.array_equal(actual,expected))
    # Real widgets, hidden root: verify controls merge instead of overwriting.
    import tkinter as tk
    from tkinter import ttk
    from types import SimpleNamespace
    from workbench_app import App
    from filter_catalog import RICOH
    root=tk.Tk();root.withdraw()
    try:
        changes=[]
        ui=SimpleNamespace(native_control_body=ttk.Frame(root),control_widgets={},look=tk.StringVar(root,value='CinemaGreen'),
            factory={},generation=1,brand=tk.StringVar(root,value=RICOH),balance_pad=None,tone_panel=None,
            overrides={},reset_controls_button=ttk.Button(root),controls_message=tk.StringVar(root),settings_changed=lambda:changes.append(1))
        ui.control_changed=lambda name:App.control_changed(ui,name)
        App.build_controls(ui)
        check('mode-specific widgets',set(ui.control_widgets)==set(control_values('CinemaGreen')))
        for name,value in (('tone',2),('contrast',-3),('saturation',1)):
            ui.control_widgets[name][0].set(value);ui.control_changed(name)
        check('widgets retain previous adjustments',ui.overrides['CinemaGreen']=={'tone':2,'contrast':-3,'saturation':1})
        ui.look.set('HardMonotone');App.build_controls(ui)
        check('mono widgets change with mode',set(ui.control_widgets)=={'key','contrast','highlight','shadow'})
    finally:root.destroy()
    from photo_edit import default_state,normalize,manual_state
    state=default_state();state['exposure']=.7;state['balance']=[3,-2]
    state['filter'].update(brand=RICOH,look='CinemaGreen',controls=controls)
    state=normalize(state);previous=manual_state(state)
    state['filter'].update(look='CrossProcess',controls={'tone':2,'key':-1})
    check('manual adjustments survive look change',manual_state(normalize(state))==previous)
    report={'checks':checks,'passed':len(checks),'native_loaded':library() is not None,'gpu':gpu,
            'scope':'colour math, input routing, hidden controls; camera hardware pixel validation pending'}
    (folder/'gr4-controls.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':len(checks),'gpu':gpu.get('device'),'report':str(folder/'gr4-controls.json')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    main(sys.argv[1])
