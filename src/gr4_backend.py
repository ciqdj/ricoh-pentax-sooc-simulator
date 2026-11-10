"""R2 photo adapter: shared format/RAW support, full-size exports, no metadata."""
from pathlib import Path
import json
import threading
import numpy as np
import tifffile
from PIL import Image
from photo_formats import read_photo,ICC,encode_png,resolve_format,RASTER_EXT
from color_math import srgb_encode
from native_backend import RenderCancelled
from gr4_renderer import render,validate_controls,control_values

_ENCODE16=np.floor(np.clip(srgb_encode(np.arange(65536,dtype=np.float64)/65535),0,1)*65535+.5).astype(np.uint16)


class GRSession:
    def __init__(self,source,preview_only=False,retain_full=False):
        self.source=Path(source);self.preview_only=preview_only;self.lock=threading.Lock()
        self.base=None;self.alpha=None;self.decode_count=0;self.size=None
        self.retain_full=retain_full;self.full_base=None;self.full_alpha=None
        self.linear_base=None;self.full_linear=None;self.input_encoding='srgb'

    def prepare(self,cancelled=None,progress=None):
        with self.lock:
            if cancelled and cancelled():raise RenderCancelled()
            if self.base is None:
                if progress:progress('读取理光效果的输入照片')
                # Decode full RAW once; both preview and export use the same
                # input preparation. Preview resizing affects display only.
                photo=read_photo(self.source)
                self.size=(photo.rgb.shape[1],photo.rgb.shape[0]);maximum=np.iinfo(photo.rgb.dtype).max
                if photo.linear and photo.rgb.dtype==np.uint16:
                    self.input_encoding='linear';self.linear_base=photo.rgb
                    base=np.empty_like(photo.rgb)
                    for row in range(0,len(base),128):
                        if cancelled and cancelled():raise RenderCancelled()
                        base[row:row+128]=_ENCODE16[photo.rgb[row:row+128]]
                elif not photo.linear:base=photo.rgb.astype(np.uint16)*257 if photo.rgb.dtype==np.uint8 else photo.rgb.copy()
                else:raise ValueError('Unexpected linear input precision')
                alpha=photo.alpha
                if alpha is not None:alpha=(alpha.astype(np.uint16)*257 if alpha.dtype==np.uint8 else alpha.copy())
                if self.retain_full:self.full_base=base;self.full_alpha=alpha
                if self.retain_full:self.full_linear=self.linear_base
                if self.preview_only:
                    w,h=self.size;scale=min(1,1600/w,1200/h);size=(max(1,round(w*scale)),max(1,round(h*scale)))
                    if scale<1:
                        if self.linear_base is not None:
                            from editor_pipeline import resize_rgb16
                            self.linear_base=resize_rgb16(self.linear_base,alpha,size)[0]
                        base=np.stack([np.floor(np.array(Image.fromarray(base[...,c].astype(np.float32)).resize(size,Image.Resampling.BILINEAR))+.5).astype(np.uint16) for c in range(3)],axis=-1)
                        if alpha is not None:alpha=np.floor(np.array(Image.fromarray(alpha.astype(np.float32)).resize(size,Image.Resampling.BILINEAR))+.5).astype(np.uint16)
                self.base=base;self.alpha=alpha;self.decode_count+=1
        return self.base,self.alpha

    def process(self,key,exposure=0,controls=None,amount=1,cancelled=None,progress=None,*,allow_gpu=True):
        tone=validate_controls(key,controls)
        base,alpha=self.prepare(cancelled,progress)
        source=self.linear_base if self.linear_base is not None else base
        result=render(source,key,controls,exposure,amount,cancelled,progress,allow_gpu=allow_gpu,input_encoding=self.input_encoding)
        report={'look':key,'brand':'ricoh_gr4','input_route':'gr4_'+self.input_encoding,'pipeline':'GR4 colour stages R3',
                'original_dcu_pipeline':False,'complete_original_gr4_engine':False,
                'native_width':self.size[0],'native_height':self.size[1],
                'preview_only':self.preview_only,'preview_width':base.shape[1],'preview_height':base.shape[0],
                'tone':tone,'exposure_ev':exposure,'user_amount_percent':amount*100,
                'personal_metadata_copied':False,'raw_decoder':'LibRaw' if self.source.suffix.lower() not in RASTER_EXT else None,
                'input_colour_space':'sRGB','input_encoding':self.input_encoding,
                'factory_defaults':control_values(key),'effective_controls':control_values(key,controls)}
        return result,alpha,report


def save(source,destination,key,exposure=0,jpeg_quality=98,bit_depth=16,controls=None,amount=1,progress=None,cancelled=None,post_balance=(0,0),post_settings=None):
    source=Path(source).resolve();destination=Path(destination).resolve();sidecar=destination.with_name(destination.name+'.json')
    if source==destination or destination.exists() or sidecar.exists():raise ValueError('输出必须是新文件，不覆盖原片')
    if destination.suffix.lower() not in RASTER_EXT:raise ValueError('不支持的输出格式')
    if bit_depth not in (8,16) or not 80<=jpeg_quality<=100:raise ValueError('导出设置无效')
    rgb,alpha,report=GRSession(source).process(key,exposure,controls,amount,cancelled,progress)
    from postprocess import process as post_process,annotate
    rgb=post_process(rgb,post_balance,post_settings,cancelled,progress,scale=1.0,alpha=alpha)
    annotate(report,post_balance,post_settings)
    fmt=resolve_format(destination);bits=bit_depth if fmt in ('png','tiff') else 8
    keep_alpha=alpha is not None and fmt in ('png','tiff','webp')
    if alpha is not None and not keep_alpha:
        a=alpha.astype(np.float64)/65535
        rgb=np.floor(rgb*a[...,None]+65535*(1-a[...,None])+.5).astype(np.uint16)
    if bits==8:rgb=((rgb.astype(np.uint32)+128)//257).astype(np.uint8)
    if keep_alpha:
        channel=alpha if bits==16 else ((alpha.astype(np.uint32)+128)//257).astype(np.uint8)
        rgb=np.concatenate([rgb,channel[...,None]],axis=2)
    destination.parent.mkdir(parents=True,exist_ok=True);created=False;sidecar_created=False
    try:
        with destination.open('xb') as out:
            created=True
            if fmt=='tiff':tifffile.imwrite(out,rgb,photometric='rgb',compression='deflate',iccprofile=ICC,extrasamples='unassalpha' if keep_alpha else None,metadata=None,software=None)
            elif fmt=='png':out.write(encode_png(rgb,b''))
            elif fmt=='jpeg':Image.fromarray(rgb).save(out,'JPEG',quality=jpeg_quality,subsampling=0,icc_profile=ICC)
            elif fmt=='webp':Image.fromarray(rgb).save(out,'WEBP',lossless=True,exact=True,icc_profile=ICC)
            else:Image.fromarray(rgb).save(out,'BMP')
        report.update(output_format=fmt,output_bits=bits,alpha_retained=keep_alpha,lossless_encoding=fmt!='jpeg')
        with sidecar.open('x',encoding='utf-8') as out:sidecar_created=True;json.dump(report,out,ensure_ascii=False,indent=2)
        return report
    except Exception:
        if created:destination.unlink(missing_ok=True)
        if sidecar_created:sidecar.unlink(missing_ok=True)
        raise
