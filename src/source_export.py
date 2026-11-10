"""One photo-owned input denoise, followed by the selected export renderer."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
from photo_denoise import key_for, working_input
from postprocess import validate


def save(store,renderer,source,destination,key,exposure,quality,bits,controls,amount,
         progress=None,cancelled=None,balance=(0,0),settings=None,backend_mode='auto',prepared_result=None):
    destination=Path(destination).resolve()
    sidecar=destination.with_name(destination.name+'.json')
    if destination.exists() or sidecar.exists():
        raise ValueError('输出必须是新文件，不覆盖原片')
    settings=validate(settings)
    result=None
    if settings['denoise']:
        expected=key_for(source,settings['denoise_model'],backend_mode)
        if prepared_result is not None and prepared_result.key==expected:
            result=prepared_result
        else:
            result=store.get_or_compute(source,settings['denoise_model'],backend_mode,
                                        cancelled=cancelled,progress=progress,supersede=False)
    context=working_input(result,settings['denoise'],cancelled) if result is not None else nullcontext(source)
    with context as working:
        report=renderer(working,destination,key,exposure,quality,bits,controls,amount,progress,
                        cancelled=cancelled,post_balance=balance,post_settings=dict(settings,denoise=0.))
    if result is not None:
        report['pre_filter_denoise']={'model':settings['denoise_model'],
            'result_amount_percent':settings['denoise'], 'input':'full_resolution_sRGB_RGB16',
            'working_copy':'lossless_RGB16', 'original_preserved':True,
            'integration':'input_before_filter', 'camera_native_raw_denoise':False}
        pending=None
        try:
            data=json.dumps(report,ensure_ascii=False,indent=2)
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=sidecar.parent,
                    prefix='.export-',suffix='.tmp',delete=False) as handle:
                pending=Path(handle.name);handle.write(data)
            os.replace(pending,sidecar)
        except BaseException:
            # The successful renderer owns these two newly-created files.
            # Keep the same all-or-failed contract if annotation cannot commit.
            for owned in (pending,sidecar,destination):
                if owned is not None:
                    try:owned.unlink(missing_ok=True)
                    except OSError:pass
            raise
    return report
