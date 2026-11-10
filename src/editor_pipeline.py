"""One photo pipeline: input denoise -> manual base -> optional selected look.

History stores parameters; this worker owns bounded, disposable pixel caches.
Interactive reductions are display-only. Export always requests full RGB16.
"""
from collections import OrderedDict
from pathlib import Path
import json,threading
import numpy as np
from PIL import Image
from photo_edit import normalize,manual_state,fingerprint,default_state
from postprocess import process,blend_clean
from photo_denoise import identity_for,key_for
from native_backend import RenderCancelled,RenderSession,LOOKS,mix_amount
from filter_catalog import PENTAX,RICOH,CUSTOM,CREATIVE_LOOKS


def _check(cancelled):
    if cancelled and cancelled():raise RenderCancelled()


def native_available(state):
    selected=state.get('filter',{})
    return selected.get('brand')==PENTAX and selected.get('look') in LOOKS


def resize_rgb16(rgb,alpha=None,maxsize=None):
    """Bilinear preview sampling preserves 16-bit channels and straight alpha."""
    height,width=rgb.shape[:2]
    if maxsize is None:return rgb,alpha,1.
    if len(maxsize)!=2 or any(type(v) is not int or v<=0 for v in maxsize):
        raise ValueError('预览尺寸无效')
    ratio=min(1.,maxsize[0]/width,maxsize[1]/height)
    if ratio==1:return rgb,alpha,1.
    size=(max(1,round(width*ratio)),max(1,round(height*ratio)))
    # Float32 represents every uint16 value exactly. Avoid quantizing to 8-bit
    # before applying any colour controls, including in temporary previews.
    def channel(a):
        out=np.asarray(Image.fromarray(a.astype(np.float32)).resize(size,Image.Resampling.BILINEAR))
        return np.rint(np.clip(out,0,65535)).astype(np.uint16)
    if alpha is None:
        result=np.stack([channel(rgb[...,c]) for c in range(3)],axis=-1)
        return result,None,min(size[0]/width,size[1]/height)
    # Premultiply only for resampling; colour processing remains straight RGB.
    a=alpha.astype(np.float32)/65535
    res_a=np.asarray(Image.fromarray(a).resize(size,Image.Resampling.BILINEAR))
    result=np.empty((size[1],size[0],3),np.uint16)
    for c in range(3):
        premul=np.asarray(Image.fromarray(rgb[...,c].astype(np.float32)*a).resize(size,Image.Resampling.BILINEAR))
        straight=np.divide(premul,res_a,out=np.zeros_like(premul),where=res_a>0)
        result[...,c]=np.rint(np.clip(straight,0,65535)).astype(np.uint16)
    return result,np.rint(np.clip(res_a,0,1)*65535).astype(np.uint16),min(size[0]/width,size[1]/height)


def apply_manual(rgb,state,alpha=None,scale=1.,cancelled=None,progress=None,cache=None):
    """Apply explicit photo edits once, before the chosen look, at RGB16."""
    settings=manual_state(state);ev=settings['exposure']
    _check(cancelled)
    def cached(name,key,compute):
        previous=cache.get(name) if cache is not None else None
        if previous is not None and previous[0]==key:return previous[1]
        value=compute();_check(cancelled)
        if cache is not None:cache[name]=(key,value)
        return value
    original=rgb
    def exposure():
        if not ev:return original
        if progress:progress('调整曝光')
        x=np.arange(65536,dtype=np.float64)/65535
        linear=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)*(2.**ev)
        encoded=np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055)
        table=np.rint(np.clip(encoded,0,1)*65535).astype(np.uint16)
        adjusted=np.empty_like(original)
        for y in range(0,len(original),128):
            _check(cancelled);adjusted[y:y+128]=table[original[y:y+128]]
        return adjusted
    exposure_key=(id(original),ev)
    rgb=cached('exposure',exposure_key,exposure)
    if progress:progress('应用影调和颜色调整')
    if cache is None:
        rgb=process(rgb,settings['balance'],settings['post'],cancelled,progress,scale=scale,alpha=alpha)
    else:
        # Fit-only caches retain the latest exact result of each preceding
        # stage. Curve movement does not rerun unchanged spatial/HSL work.
        # Use the same process functions and RGB16 rounding as final export.
        basic_key=(exposure_key,id(alpha),scale,fingerprint(settings['post']['basic']))
        rgb=cached('basic',basic_key,lambda:process(rgb,(0,0),{'basic':settings['post']['basic']},
            cancelled,progress,scale=scale,alpha=alpha))
        mixer_key=(basic_key,fingerprint(settings['post']['mixer']))
        rgb=cached('mixer',mixer_key,lambda:process(rgb,(0,0),{'mixer':settings['post']['mixer']},cancelled,progress))
        balance_key=(mixer_key,tuple(settings['balance']))
        rgb=cached('balance',balance_key,lambda:process(rgb,settings['balance'],{},cancelled,progress))
        rgb=process(rgb,(0,0),{'tone':settings['post']['tone']},cancelled,progress)
    _check(cancelled)
    return rgb


class EditorSession:
    """Single-worker session. Two fit bases and one full base at most.

    Native process workspaces live only as long as their exact manual base.
    AI result objects supplied by the UI are leased during each render; their
    model is never rerun because of colour edits or selecting another look.
    """
    def __init__(self,path,store,library=None):
        self.path=Path(path);self.store=store;self.library=library
        self.identity=identity_for(path);self.prepared=None;self.closed=False
        self._bases=OrderedDict();self._samples=OrderedDict();self._manual_caches=OrderedDict();self._lock=threading.RLock()
        self.decode_count=0;self._source_token=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def close(self):
        with self._lock:
            self.closed=True
            for entry in self._bases.values():
                if entry.get('native') is not None:entry['native'].close()
            self._bases.clear();self._samples.clear();self._manual_caches.clear();self.prepared=None

    def _prepare(self,cancelled,progress):
        _check(cancelled)
        if self.closed:raise RenderCancelled()
        if identity_for(self.path)!=self.identity:raise ValueError('原片已更改，请重新打开照片')
        if self.prepared is None:
            self.prepared=self.store.prepare(self.path,cancelled=cancelled,progress=progress)
            self.decode_count+=1
        return self.prepared

    def _base(self,state,result,maxsize,cancelled,progress):
        prepared=self._prepare(cancelled,progress)
        dn=state['denoise'];valid=(dn['kind']=='ai' and result is not None and
            result.key.identity==self.identity and result.key.model==dn['model'] and
            result.rgb.shape==prepared.rgb.shape and result.rgb.dtype==np.uint16)
        source_id=(id(result),dn['amount']) if valid else None
        if source_id!=self._source_token:
            # A different denoise result is a different input, not another
            # history-sized image cache. Release old leases/workspaces now.
            for previous in self._bases.values():
                if previous['native'] is not None:previous['native'].close()
            self._bases.clear();self._samples.clear();self._manual_caches.clear();self._source_token=source_id
        size=None if maxsize is None else tuple(maxsize)
        sample_key=(source_id,size)
        sample=self._samples.get(sample_key)
        if sample is None:
            _check(cancelled)
            original,alpha,scale=resize_rgb16(prepared.rgb,prepared.alpha,maxsize)
            if valid:
                clean,_,_=resize_rgb16(result.rgb,prepared.alpha,maxsize)
                rgb=blend_clean(original,clean,dn['amount']/100,cancelled)
            else:rgb=original
            sample=(original,rgb,alpha,scale,result if valid else None)
            self._samples[sample_key]=sample
            for old in tuple(self._samples):
                if old!=sample_key and old[1] is None:self._samples.pop(old)
            while len(self._samples)>3:self._samples.popitem(last=False)
        else:self._samples.move_to_end(sample_key)
        original,rgb,alpha,scale,_lease=sample
        manual=manual_state(state);key=(sample_key,fingerprint(manual))
        entry=self._bases.get(key)
        if entry is None:
            stage_cache=None
            if size is not None:
                stage_cache=self._manual_caches.setdefault(sample_key,{})
                self._manual_caches.move_to_end(sample_key)
                while len(self._manual_caches)>2:self._manual_caches.popitem(last=False)
            base=apply_manual(rgb,state,alpha,scale,cancelled,progress,stage_cache)
            base.flags.writeable=False
            entry={'original':original,'base':base,'alpha':alpha,'scale':scale,'native':None}
            self._bases[key]=entry
            # Keep one full-size manual result, irrespective of its history.
            for old in tuple(self._bases):
                if old!=key and old[0][1] is None:
                    removed=self._bases.pop(old)
                    if removed['native'] is not None:removed['native'].close()
            while len(self._bases)>3:
                _,removed=self._bases.popitem(last=False)
                if removed['native'] is not None:removed['native'].close()
        else:self._bases.move_to_end(key)
        return prepared,entry,bool(dn['kind']=='ai' and dn['amount']>0 and not valid)

    def render(self,state,*,maxsize=None,result=None,cancelled=None,progress=None):
        state=normalize(state)
        with self._lock:
            if state['denoise']['kind']=='native':
                if not native_available(state):raise ValueError('DCU 降噪需要选择宾得原厂滤镜')
                if state['filter']['amount']!=1.:raise ValueError('DCU 降噪使用原厂浓淡 100%，避免混回未降噪图像')
            prepared,entry,pending=self._base(state,result,maxsize,cancelled,progress)
            base=entry['base'];alpha16=entry['alpha']
            alpha=None if alpha16 is None else alpha16.astype(np.float64)/65535
            selected=state['filter'];brand,key,amount=selected['brand'],selected['look'],selected['amount']
            report={'look':key or '', 'brand':brand,'input_route':'edited_rgb16',
                'pipeline':'input_denoise_then_manual_base_then_optional_filter',
                'original_dcu_pipeline':False,'input_colour_space':'sRGB',
                'personal_metadata_copied':False}
            before=entry['original']
            if not key:
                rgb=base;report['input_route']='original_rgb16'
            elif brand==PENTAX and key in LOOKS:
                controls=dict(selected['controls'])
                if state['denoise']['kind']=='native':controls.update(state['denoise']['native'])
                # Untouched PEF full-size retains its original native route.
                # Edited or AI-denoised sources are explicitly RGB inputs.
                native_pef=(maxsize is None and self.path.suffix.lower()=='.pef' and
                            state['denoise']['kind']!='ai' and
                            manual_state(state)==manual_state(default_state()))
                if entry['native'] is not None and entry.get('native_pef')!=native_pef:
                    entry['native'].close();entry['native']=None
                if entry['native'] is None:
                    entry['native']=(RenderSession(self.path) if native_pef else
                        RenderSession.from_rgb(base,alpha,prepared.notes))
                    entry['native_pef']=native_pef
                native=entry['native']
                rgb,returned_alpha,native_report=native.render(key,0.,progress,cancelled,controls)
                report.update(native_report)
                if native_pef:
                    native_base,native_alpha=native.baseline(key,progress,cancelled)
                    before=native_base;alpha=native_alpha;base=native_base
                elif returned_alpha is not None:alpha=returned_alpha
                if amount!=1.:rgb=mix_amount(base,rgb,amount)
                report['original_dcu_pipeline']=True
                report['native_custom_noise']=state['denoise']['kind']=='native'
            elif brand==RICOH:
                from gr4_renderer import render,control_values
                settings=control_values(key,selected['controls'])
                linear_input=(prepared.linear_rgb is not None and
                    manual_state(state)==manual_state(default_state()) and
                    not (state['denoise']['kind']=='ai' and state['denoise']['amount']>0 and not pending))
                encoding='linear' if linear_input else 'srgb'
                if linear_input:
                    if 'linear' not in entry:
                        entry['linear']=resize_rgb16(prepared.linear_rgb,prepared.alpha,maxsize)[0]
                    source=entry['linear']
                else:source=base
                rgb=render(source,key,selected['controls'],0.,amount,cancelled,progress,input_encoding=encoding)
                report.update(input_route='gr4_'+encoding,complete_original_gr4_engine=False,
                    pipeline='GR4 colour stages R3',factory_defaults=control_values(key),
                    effective_controls=settings,tone=settings.get('tone',0),input_encoding=encoding)
            elif brand==CUSTOM:
                from custom_lut import get_library,apply_cube
                library=self.library if self.library is not None else get_library()
                cube=library.get_cube(key,cancelled)
                rgb=apply_cube(base,cube,amount,0.,cancelled,progress)
                item=library.lookup(key)
                report.update(input_route='custom_cube_rgb16',cube_sha256=item['sha256'],
                    cube_kind=item['kind'],cube_size=item['size'])
            elif brand==PENTAX and key in CREATIVE_LOOKS:
                from process_photo import adjust_rgb
                rgb=np.empty_like(base)
                for row in range(0,len(base),128):
                    _check(cancelled)
                    value=adjust_rgb(base[row:row+128].astype(np.float64)/65535,False,key,amount,0.)
                    rgb[row:row+128]=np.floor(np.clip(value,0,1)*65535+.5).astype(np.uint16)
                report.update(input_route='legacy31_creative',pipeline='3.1 creative formula')
            else:raise ValueError('所选滤镜不可用，请重新选择')
            _check(cancelled)
            native_size=prepared.size if maxsize is not None else (rgb.shape[1],rgb.shape[0])
            report.update(native_width=native_size[0],native_height=native_size[1],
                preview_only=maxsize is not None,preview_width=rgb.shape[1],preview_height=rgb.shape[0],
                user_amount_percent=amount*100 if key else 0.,
                input_denoise_pending=pending,manual_base=manual_state(state),
                manual_position='before_filter',source_precision='RGB16')
            if state['denoise']['kind']=='ai':
                report['pre_filter_denoise']={'model':state['denoise']['model'],
                    'result_amount_percent':state['denoise']['amount'],'complete':not pending,
                    'camera_native_raw_denoise':False,'working_copy':'lossless_RGB16'}
            return before,rgb,alpha,report


def save(store,source,destination,state,jpeg_quality=98,bit_depth=16,backend_mode='auto',
         progress=None,cancelled=None,prepared_result=None,library=None):
    """Export the exact captured history state, never a downscaled display."""
    import tifffile
    from photo_formats import RASTER_EXT,resolve_format,ICC,encode_png
    destination=Path(destination).resolve();source=Path(source).resolve()
    sidecar=destination.with_name(destination.name+'.json')
    if source==destination or destination.exists() or sidecar.exists():raise ValueError('输出必须是新文件，不覆盖原片')
    if destination.suffix.lower() not in RASTER_EXT:raise ValueError('不支持的输出格式')
    if bit_depth not in (8,16) or not 80<=jpeg_quality<=100:raise ValueError('导出设置无效')
    state=normalize(state);result=None
    if state['denoise']['kind']=='ai' and state['denoise']['amount']:
        expected=key_for(source,state['denoise']['model'],backend_mode)
        if prepared_result is not None and prepared_result.key==expected:result=prepared_result
        else:
            result=store.get_or_compute(source,state['denoise']['model'],backend_mode,
                cancelled=cancelled,progress=progress,supersede=False)
    with EditorSession(source,store,library) as session:
        before,rgb,alpha,report=session.render(state,result=result,cancelled=cancelled,progress=progress)
        if state['denoise']['kind']=='ai' and state['denoise']['amount'] and report['input_denoise_pending']:
            raise RuntimeError('降噪结果与当前原片不匹配，尚未导出')
        fmt=resolve_format(destination);bits=bit_depth if fmt in ('png','tiff') else 8
        keep_alpha=alpha is not None and fmt in ('png','tiff','webp')
        if alpha is not None and not keep_alpha:
            merged=np.empty_like(rgb)
            for y in range(0,len(rgb),128):
                _check(cancelled);a=alpha[y:y+128][...,None]
                merged[y:y+128]=np.rint(rgb[y:y+128]*a+65535*(1-a)).astype(np.uint16)
            rgb=merged
        if bits==8:rgb=((rgb.astype(np.uint32)+128)//257).astype(np.uint8)
        if keep_alpha:rgb=np.concatenate([rgb,np.rint(alpha*(65535 if bits==16 else 255)).astype(rgb.dtype)[...,None]],axis=2)
        _check(cancelled);destination.parent.mkdir(parents=True,exist_ok=True)
        created=False;side_created=False
        try:
            with destination.open('xb') as handle:
                created=True
                if fmt=='tiff':tifffile.imwrite(handle,rgb,photometric='rgb',compression='deflate',iccprofile=ICC,
                    extrasamples='unassalpha' if keep_alpha else None,metadata=None,software=None)
                elif fmt=='png':handle.write(encode_png(rgb,b''))
                elif fmt=='jpeg':Image.fromarray(rgb).save(handle,'JPEG',quality=jpeg_quality,subsampling=0,icc_profile=ICC)
                elif fmt=='webp':Image.fromarray(rgb).save(handle,'WEBP',lossless=True,exact=True,icc_profile=ICC)
                else:Image.fromarray(rgb).save(handle,'BMP')
            _check(cancelled)
            report.update(output_format=fmt,output_bits=bits,alpha_retained=keep_alpha,
                          lossless_encoding=fmt!='jpeg',original_preserved=True)
            with sidecar.open('x',encoding='utf-8') as handle:
                side_created=True;json.dump(report,handle,ensure_ascii=False,indent=2)
            return report
        except BaseException:
            if created:destination.unlink(missing_ok=True)
            if side_created:sidecar.unlink(missing_ok=True)
            raise
