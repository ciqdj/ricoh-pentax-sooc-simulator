"""User-enabled post-filter operations. Neutral settings leave pixels intact."""
import math
import numpy as np
from PIL import Image
from tone_tools import validate as validate_tone,apply_array as apply_tone
from color_balance import validate as validate_balance,apply_rgb16,apply_image
from basic_adjustments import validate as validate_basic,neutral as basic_neutral,apply as apply_basic
from color_mixer import validate as validate_mixer,neutral as mixer_neutral,apply as apply_mixer


def validate(value=None):
    value=value or {}
    if not isinstance(value,dict) or set(value)-{'tone','basic','mixer','denoise','denoise_model','noise_level'}:raise ValueError('后置调整格式无效')
    amount=value.get('denoise',0)
    if type(amount) not in (int,float) or not math.isfinite(amount) or not 0<=amount<=100:raise ValueError('降噪强度应为 0–100')
    model=value.get('denoise_model','dcu');level=value.get('noise_level',15)
    # Preserve the other edits in presets saved by the unreleased fast-model
    # build, but require an explicit opt-in before running the new model.
    if model in ('ffdnet','scunet'):model='dcu';amount=0
    if model not in ('dcu','nafnet','restormer') or type(level) not in (int,float) or not math.isfinite(level) or not 1<=level<=50:raise ValueError('降噪方式或兼容参数无效')
    return {'tone':validate_tone(value.get('tone')),'basic':validate_basic(value.get('basic')),
            'mixer':validate_mixer(value.get('mixer')),
            'denoise':float(amount),'denoise_model':model,'noise_level':float(level)}


def blend_clean(rgb,clean,amount,cancelled=None):
    """Blend an already completed full-image denoise result without rerunning.

    The same rounding is used by the export denoiser. Small row blocks avoid a
    full-image float64 temporary when a high-resolution preview is requested.
    """
    if rgb.shape!=clean.shape or rgb.dtype!=clean.dtype or rgb.dtype not in (np.uint8,np.uint16):raise ValueError('降噪缓存与当前图像不匹配')
    if not math.isfinite(amount) or not 0<=amount<=1:raise ValueError('降噪混合强度无效')
    if amount==0:return rgb
    if amount==1:return clean
    result=np.empty_like(rgb)
    for y in range(0,rgb.shape[0],64):
        if cancelled and cancelled():
            from native_backend import RenderCancelled
            raise RenderCancelled()
        a=rgb[y:y+64].astype(np.float64);b=clean[y:y+64].astype(np.float64)
        result[y:y+64]=np.floor(np.clip(a+(b-a)*amount,0,np.iinfo(rgb.dtype).max)+.5).astype(rgb.dtype)
    return result


def process(rgb,balance=(0,0),settings=None,cancelled=None,progress=None,scale=1.0,alpha=None):
    """Process source precision; ``scale`` is display pixels per source pixel.

    Full-size exports and 100% crops use scale=1. Alpha is supplied only for
    spatial operations and remains independent of every colour adjustment.
    """
    settings=validate(settings);balance=validate_balance(balance)
    if settings['denoise']:
        from denoise import apply as denoise
        rgb=denoise(rgb,settings['denoise']/100,cancelled,progress,settings['denoise_model'],settings['noise_level'])
    if not basic_neutral(settings['basic']):
        rgb=apply_basic(rgb,settings['basic'],cancelled,scale=scale,alpha=alpha)
    if not mixer_neutral(settings['mixer']):
        rgb=apply_mixer(rgb,settings['mixer'],cancelled)
    if any(balance):
        rgb=apply_rgb16(rgb,balance,cancelled) if rgb.dtype==np.uint16 else np.asarray(apply_image(Image.fromarray(rgb),balance))
    return apply_tone(rgb,settings['tone'],cancelled)


def preview(image,balance=(0,0),settings=None,cancelled=None,progress=None,scale=1.0,alpha=None):
    settings=validate(settings)
    from tone_tools import neutral
    if not any(balance) and not settings['denoise'] and neutral(settings['tone']) and basic_neutral(settings['basic']) and mixer_neutral(settings['mixer']):return image
    # Keep an image's own transparency. An explicit alpha array may also be
    # used with straight RGB, without changing the caller's display mode.
    image_alpha=image.getchannel('A') if 'A' in image.getbands() else None
    if alpha is None and image_alpha is not None:alpha=np.asarray(image_alpha)
    result=Image.fromarray(process(np.asarray(image.convert('RGB')),balance,settings,cancelled,progress,scale,alpha))
    if image_alpha is not None:result.putalpha(image_alpha)
    return result


def annotate(report,balance,settings):
    settings=validate(settings)
    from tone_tools import neutral
    if any(balance):report['post_filter_balance_lab']=list(balance)
    if not basic_neutral(settings['basic']):
        report['post_filter_basic']={'values':settings['basic'],'camera_native':False,
                                     'integration':'manual_post_filter','spatial_radius_reference':'source_pixels'}
    if not mixer_neutral(settings['mixer']):
        report['post_filter_mixer']={'values':settings['mixer'],'camera_native':False,'integration':'manual_post_filter'}
    if not neutral(settings['tone']):report['post_filter_tone']=settings['tone']
    if settings['denoise']:
        dcu=settings['denoise_model']=='dcu'
        report['post_filter_denoise']={'model':settings['denoise_model'],'amount_percent':settings['denoise'],
            'camera_native':False,'ai':not dcu,'integration':'independent_post_filter'}
        if dcu:report['post_filter_denoise'].update(module='DCU NoiseCut',module_settings='original_defaults')
