"""Restore the exact 3.1 creative formula with current private-metadata handling."""
from pathlib import Path
import json
import threading
import numpy as np
import tifffile
from PIL import Image
from engine import srgb_encode
from process_photo import adjust_rgb
from photo_formats import read_photo, ICC, encode_png, resolve_format, RASTER_EXT, RAW_EXT
from filter_catalog import CREATIVE_LOOKS
from native_backend import RenderCancelled


class CreativeSession:
    def __init__(self,source):
        self.source=source;self.photo=None;self.base=None;self.alpha=None;self.lock=threading.Lock()

    def prepare(self,cancelled=None):
        with self.lock:
            if cancelled and cancelled():raise RenderCancelled()
            if self.photo is None:
                photo=read_photo(self.source)
                a=photo.rgb;maximum=np.iinfo(a.dtype).max
                base=np.empty(a.shape,np.uint16)
                for row in range(0,len(a),128):
                    if cancelled and cancelled():raise RenderCancelled()
                    block=a[row:row+128].astype(np.float64)/maximum
                    if photo.linear:block=srgb_encode(block)
                    base[row:row+128]=np.floor(np.clip(block,0,1)*65535+.5).astype(np.uint16)
                self.alpha=photo.alpha.astype(np.float64)/maximum if photo.alpha is not None else None
                self.photo=photo;self.base=base
        return self.base,self.alpha

    def render(self,key,amount=1,exposure=0,cancelled=None,progress=None):
        if key not in CREATIVE_LOOKS:raise ValueError('未知创意滤镜')
        self.prepare(cancelled)
        photo=self.photo;a=photo.rgb;maximum=np.iinfo(a.dtype).max
        out=np.empty(a.shape,np.uint16)
        for row in range(0,len(a),128):
            if cancelled and cancelled():raise RenderCancelled()
            block=adjust_rgb(a[row:row+128].astype(np.float64)/maximum,photo.linear,key,amount,exposure)
            out[row:row+128]=np.floor(np.clip(block,0,1)*65535+.5).astype(np.uint16)
            if progress:progress('处理 3.1 创意滤镜')
        report={'look':key,'input_route':'legacy31_creative','native_width':a.shape[1],'native_height':a.shape[0],
                'pipeline':'3.1 creative formula','original_dcu_pipeline':False,
                'user_amount_percent':amount*100,'exposure_ev':exposure,'personal_metadata_copied':False}
        return out,self.alpha,report


class CreativePreview:
    """Match 3.1's half-RAW/1200x900 preview preparation, keep export full-size."""
    def __init__(self,source):
        self.source=Path(source);self.data=None;self.lock=threading.Lock()

    def prepare(self,cancelled=None):
        with self.lock:
            if cancelled and cancelled():raise RenderCancelled()
            if self.data is None:
                photo=read_photo(self.source,half_size=self.source.suffix.lower() in RAW_EXT)
                maximum=np.iinfo(photo.rgb.dtype).max
                rgb=photo.rgb.astype(np.float32)/maximum
                h,w=rgb.shape[:2];ratio=min(1,1200/w,900/h)
                size=(max(1,round(w*ratio)),max(1,round(h*ratio)))
                if ratio<1:rgb=np.stack([np.array(Image.fromarray(rgb[...,i]).resize(size,Image.Resampling.BILINEAR)) for i in range(3)],axis=-1)
                alpha=None
                if photo.alpha is not None:
                    alpha=photo.alpha.astype(np.float32)/maximum
                    if ratio<1:alpha=np.array(Image.fromarray(alpha).resize(size,Image.Resampling.BILINEAR))
                if cancelled and cancelled():raise RenderCancelled()
                self.data=(rgb,photo.linear,alpha)
        return self.data

    def render(self,key,amount=1,exposure=0,cancelled=None):
        rgb,linear,alpha=self.prepare(cancelled)
        before=srgb_encode(rgb) if linear else rgb
        after=adjust_rgb(rgb,linear,key,amount,exposure)
        if cancelled and cancelled():raise RenderCancelled()
        report={'look':key,'input_route':'legacy31_creative','preview_only':True,
                'preview_width':rgb.shape[1],'preview_height':rgb.shape[0],
                'pipeline':'3.1 creative formula','original_dcu_pipeline':False,
                'user_amount_percent':amount*100,'exposure_ev':exposure,'personal_metadata_copied':False}
        return before,after,alpha,report


def save(source,destination,key,exposure=0,jpeg_quality=98,bit_depth=16,controls=None,amount=1,progress=None,cancelled=None,post_balance=(0,0),post_settings=None):
    source=Path(source).resolve();destination=Path(destination).resolve()
    sidecar=destination.with_name(destination.name+'.json')
    if source==destination or destination.exists() or sidecar.exists():raise ValueError('输出必须是新文件，不覆盖原片')
    if destination.suffix.lower() not in RASTER_EXT:raise ValueError('不支持的输出格式')
    if bit_depth not in (8,16) or not 80<=jpeg_quality<=100:raise ValueError('导出设置无效')
    if controls:raise ValueError('3.1 创意滤镜不使用原厂专属参数')
    session=CreativeSession(source)
    rgb,alpha,report=session.render(key,amount,exposure,cancelled,progress)
    from postprocess import process as post_process,annotate
    rgb=post_process(rgb,post_balance,post_settings,cancelled,progress,scale=1.0,alpha=alpha)
    annotate(report,post_balance,post_settings)
    fmt=resolve_format(destination);bits=bit_depth if fmt in ('png','tiff') else 8
    keep_alpha=alpha is not None and fmt in ('png','tiff','webp')
    if alpha is not None and not keep_alpha:rgb=np.rint(rgb.astype(np.float64)*alpha[...,None]+65535*(1-alpha[...,None])).astype(np.uint16)
    if bits==8:rgb=((rgb.astype(np.uint32)+128)//257).astype(np.uint8)
    if keep_alpha:rgb=np.concatenate([rgb,np.rint(alpha*(65535 if bits==16 else 255)).astype(rgb.dtype)[...,None]],axis=2)
    destination.parent.mkdir(parents=True,exist_ok=True);created=False;sidecar_created=False
    try:
        with destination.open('xb') as out:
            created=True
            if fmt=='tiff':tifffile.imwrite(out,rgb,photometric='rgb',compression='deflate',iccprofile=ICC,extrasamples='unassalpha' if keep_alpha else None,metadata=None,software=None)
            elif fmt=='png':out.write(encode_png(rgb,b''))
            elif fmt=='jpeg':Image.fromarray(rgb).save(out,'JPEG',quality=jpeg_quality,subsampling=0,icc_profile=ICC)
            elif fmt=='webp':Image.fromarray(rgb).save(out,'WEBP',lossless=True,exact=True,icc_profile=ICC)
            else:Image.fromarray(rgb).save(out,'BMP')
        report.update({'output_format':fmt,'output_bits':bits,'alpha_retained':keep_alpha,'lossless_encoding':fmt!='jpeg'})
        with sidecar.open('x',encoding='utf-8') as out:sidecar_created=True;json.dump(report,out,ensure_ascii=False,indent=2)
        return report
    except Exception:
        if created:destination.unlink(missing_ok=True)
        if sidecar_created:sidecar.unlink(missing_ok=True)
        raise
