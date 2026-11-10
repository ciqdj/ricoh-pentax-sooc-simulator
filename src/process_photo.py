"""Photo processing with explicit format, bit depth, metadata and alpha handling."""
from pathlib import Path
from dataclasses import dataclass
import argparse, hashlib, io, json, math, struct, zlib
import numpy as np
from PIL import Image, ImageCms
import rawpy, tifffile, imagecodecs
from engine import apply, LOOKS, srgb_encode, srgb_decode

RAW_EXT={'.arw','.pef','.dng','.nef','.nrw','.cr2','.cr3','.crw','.raf','.rw2','.orf','.srw','.rwl','.3fr','.iiq','.kdc','.dcr','.mos','.mrw','.x3f'}
RASTER_EXT={'.jpg','.jpeg','.jpe','.jfif','.png','.tif','.tiff','.webp','.bmp'}
PHOTO_EXT=RAW_EXT|RASTER_EXT
SRGB=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
ICC=SRGB.tobytes()
FORMAT_CHOICES={'跟随输入格式':'auto','JPEG':'jpeg','PNG':'png','TIFF':'tiff','WebP':'webp','BMP':'bmp'}
FORMAT_EXT={'jpeg':'.jpg','png':'.png','tiff':'.tif','webp':'.webp','bmp':'.bmp'}

@dataclass
class Photo:
    rgb: np.ndarray
    linear: bool
    notes: list
    alpha: np.ndarray|None=None
    exif: bytes=b''
    metadata: dict|None=None

def orient(array, orientation):
    if orientation==2:return array[:,::-1]
    if orientation==3:return array[::-1,::-1]
    if orientation==4:return array[::-1]
    if orientation==5:return array.swapaxes(0,1)
    if orientation==6:return np.rot90(array,3)
    if orientation==7:return array.swapaxes(0,1)[::-1,::-1]
    if orientation==8:return np.rot90(array,1)
    return array

def metadata_from(path):
    """Retain standard photographic EXIF; discard stale layout pointers/MakerNotes."""
    info={}; output=Image.Exif(); profile=None; orientation=1
    try:
        with Image.open(path) as im:
            profile=im.info.get('icc_profile'); src=im.getexif()
            orientation=int(src.get(274,1))
            for tag in (270,271,272,305,306,315,33432):
                if tag in src:output[tag]=src[tag]
            exif=dict(src.get_ifd(34665)) if 34665 in src else {}
            for tag in (37500,40965,513,514):exif.pop(tag,None)
            if exif:output[34665]=exif
            if 34853 in src:output[34853]=dict(src.get_ifd(34853))
            names={271:'camera_make',272:'camera_model',306:'date_time'}
            info={name:str(src[tag]) for tag,name in names.items() if tag in src}
            names={33434:'exposure_time',33437:'f_number',34855:'iso',36867:'date_time_original',37386:'focal_length',42036:'lens'}
            info.update({name:str(exif[tag]) for tag,name in names.items() if tag in exif})
    except (OSError,ValueError,TypeError,KeyError):
        pass
    return output,profile,orientation,info

def read_photo(path,half_size=False):
    path=Path(path); notes=[]; linear=False; alpha=None
    exif,profile,orientation,metadata=metadata_from(path)
    ext=path.suffix.lower()
    with path.open('rb') as stream:header=stream.read(25)
    png16=ext=='.png' and len(header)==25 and header[24]==16
    if ext in RAW_EXT:
        with rawpy.imread(str(path)) as raw:
            array=raw.postprocess(use_camera_wb=True,use_auto_wb=False,output_color=rawpy.ColorSpace.sRGB,
                output_bps=16,gamma=(1,1),no_auto_bright=True,half_size=half_size)
        linear=True;profile=None;orientation=1
        notes.append('RAW developed by LibRaw with camera white balance, not DCU5 demosaicing. RAW originals are never rewritten.')
    elif ext in {'.tif','.tiff'}:
        with tifffile.TiffFile(path) as tf:
            if len(tf.pages)!=1:raise ValueError('多页 TIFF 请先选择并导出其中一页。')
            page=tf.pages[0]
            if page.photometric not in (1,2):raise ValueError('TIFF 支持 RGB、灰度和透明通道；CMYK 等格式请先转换为 sRGB。')
            tag=page.tags.get(34675)
            if tag:profile=tag.value
            tag=page.tags.get(274);orientation=int(tag.value) if tag else 1
            array=page.asarray()
            if page.planarconfig==2:array=np.moveaxis(array,0,-1)
            associated=bool(page.extrasamples and page.extrasamples[0]==1)
        if array.ndim==2:array=np.repeat(array[...,None],3,axis=2)
        if array.dtype not in (np.uint8,np.uint16) or array.shape[-1] not in (3,4):
            raise ValueError('支持 8/16 位 RGB / RGBA TIFF；浮点 HDR TIFF 暂不支持。')
        if array.shape[-1]==4 and associated:
            mx=np.iinfo(array.dtype).max; a=array[...,3:4].astype(np.float64)
            rgb=np.divide(array[...,:3].astype(np.float64)*mx,a,out=np.zeros_like(array[...,:3],dtype=np.float64),where=a>0)
            array[...,:3]=np.rint(np.clip(rgb,0,mx)).astype(array.dtype)
            notes.append('Associated TIFF alpha converted to straight alpha.')
    elif png16:
        array=imagecodecs.png_decode(path.read_bytes())
        if array.ndim==2:array=np.repeat(array[...,None],3,axis=2)
        elif array.shape[-1]==2:array=np.concatenate([np.repeat(array[...,:1],3,axis=2),array[...,1:2]],axis=2)
    else:
        with Image.open(path) as im:
            if getattr(im,'n_frames',1)!=1:raise ValueError('动态图片请先导出一张静态照片。')
            has_alpha=im.mode in ('RGBA','LA','PA') or 'transparency' in im.info
            if profile:
                source_profile=ImageCms.ImageCmsProfile(io.BytesIO(profile))
                base=im if im.mode in ('RGB','CMYK','LAB','L') else im.convert('RGB')
                converted=ImageCms.profileToProfile(base,source_profile,SRGB,outputMode='RGB')
                if has_alpha:converted.putalpha(im.convert('RGBA').getchannel('A'))
                im=converted;profile=None
                notes.append('Embedded ICC profile converted to sRGB with LittleCMS.')
            else:im=im.convert('RGBA' if has_alpha else 'RGB')
            array=np.array(im)
    if profile:
        source_profile=ImageCms.ImageCmsProfile(io.BytesIO(profile))
        if 'srgb' not in ImageCms.getProfileDescription(source_profile).lower():
            if array.dtype==np.uint16:
                raise ValueError('这张 16 位照片不是 sRGB。为避免降成 8 位，请先在 PS 中“转换为配置文件 → sRGB IEC61966-2.1”。')
            a=array[...,3:4] if array.shape[-1]==4 else None
            rgb=np.array(ImageCms.profileToProfile(Image.fromarray(array[...,:3]),source_profile,SRGB,outputMode='RGB'))
            array=np.concatenate([rgb,a],axis=2) if a is not None else rgb
            notes.append('Embedded ICC profile converted to sRGB with LittleCMS.')
    elif not linear and not notes:notes.append('Untagged input interpreted as sRGB.')
    array=np.ascontiguousarray(orient(array,orientation))
    if array.shape[-1]==4:alpha=array[...,3].copy();array=array[...,:3].copy()
    exif[274]=1
    exif_ifd=exif.get(34665,{})
    exif_ifd.update({40961:1,40962:array.shape[1],40963:array.shape[0]});exif[34665]=exif_ifd
    return Photo(array,linear,notes,alpha,exif.tobytes(),metadata)

def load_photo(path,half_size=False):
    photo=read_photo(path,half_size)
    return photo.rgb,photo.linear,photo.notes

def adjust_rgb(rgb,linear,look,amount=1.,exposure=0.):
    if not math.isfinite(amount) or not 0<=amount<=2:raise ValueError('滤镜浓淡范围为 0–200%。')
    if not math.isfinite(exposure) or not -5<=exposure<=5:raise ValueError('曝光补偿范围为 -5 到 +5 EV。')
    if linear:base=srgb_encode(np.clip(rgb*(2**exposure),0,1))
    elif exposure:base=srgb_encode(np.clip(srgb_decode(rgb)*(2**exposure),0,1))
    else:base=rgb
    return apply(base,look,amount)

def resolve_format(source,format_name='auto'):
    if format_name!='auto':
        if format_name not in FORMAT_EXT:raise ValueError('不支持的输出格式。')
        return format_name
    ext=Path(source).suffix.lower()
    if ext in RAW_EXT:return 'tiff'
    if ext in {'.jpg','.jpeg','.jpe','.jfif'}:return 'jpeg'
    if ext in {'.tif','.tiff'}:return 'tiff'
    if ext in {'.png','.webp','.bmp'}:return ext[1:]
    raise ValueError('无法跟随此输入格式，请明确选择输出格式。')

def destination_for(source,folder,look,format_name='auto'):
    fmt=resolve_format(source,format_name);stem=Path(source).stem+'_'+look
    dest=Path(folder)/(stem+FORMAT_EXT[fmt]);n=1
    while dest.exists() or dest.with_name(dest.name+'.json').exists():
        dest=Path(folder)/(stem+f'_{n}'+FORMAT_EXT[fmt]);n+=1
    return dest

def png_chunk(kind,data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)

def encode_png(array,exif):
    encoded=imagecodecs.png_encode(np.ascontiguousarray(array),level=6)
    meta=png_chunk(b'iCCP',b'sRGB\0\0'+zlib.compress(ICC))
    if exif:meta+=png_chunk(b'eXIf',exif[6:] if exif.startswith(b'Exif\0\0') else exif)
    return encoded[:33]+meta+encoded[33:]

def process(source,dest,look='GOLD',amount=1.0,exposure=0.0,half_size=False,progress=None,
            jpeg_quality=98,bit_depth=16):
    source=Path(source).resolve();dest=Path(dest).resolve();sidecar=dest.with_name(dest.name+'.json')
    if source==dest or dest.exists() or sidecar.exists():raise ValueError('输出必须是新文件，不能覆盖原片或已有处理记录。')
    if look not in LOOKS:raise ValueError('未知风格。')
    if dest.suffix.lower() not in RASTER_EXT:raise ValueError('请选择 JPEG、PNG、TIFF、WebP 或 BMP 输出。')
    fmt=resolve_format(dest)
    if not isinstance(jpeg_quality,int) or not 80<=jpeg_quality<=100:raise ValueError('JPEG 质量范围为 80–100。')
    if bit_depth not in (8,16):raise ValueError('输出位深只能是 8 或 16。')
    adjust_rgb(np.zeros((1,3)),False,look,amount,exposure)
    photo=read_photo(source,half_size);array=photo.rgb
    maximum=float(np.iinfo(array.dtype).max);bits=bit_depth if fmt in ('tiff','png') else 8
    output_max=65535 if bits==16 else 255
    result=np.empty(array.shape,dtype=np.uint16 if bits==16 else np.uint8)
    notes=list(photo.notes)
    keep_alpha=photo.alpha is not None and fmt in ('tiff','png','webp')
    for top in range(0,array.shape[0],128):
        block=adjust_rgb(array[top:top+128].astype(np.float64)/maximum,photo.linear,look,amount,exposure)
        if photo.alpha is not None and not keep_alpha:
            a=photo.alpha[top:top+128,...,None].astype(np.float64)/maximum
            block=block*a+(1-a)
        result[top:top+128]=np.floor(np.clip(block,0,1)*output_max+.5).astype(result.dtype)
        if progress:progress(min(top+128,array.shape[0])/array.shape[0])
    if keep_alpha:
        alpha=np.rint(photo.alpha.astype(np.float64)/maximum*output_max).astype(result.dtype)
        result=np.concatenate([result,alpha[...,None]],axis=2)
    elif photo.alpha is not None:notes.append('Transparent pixels composited over white for this output format.')
    dest.parent.mkdir(parents=True,exist_ok=True)
    created=False;sidecar_created=False
    try:
        with dest.open('xb') as f:
            created=True
            if fmt=='tiff':
                tags=[];meta=photo.metadata or {}
                for key,tag in [('camera_make',271),('camera_model',272),('date_time',306)]:
                    if key in meta:tags.append((tag,'s',0,meta[key],False))
                tifffile.imwrite(f,result,photometric='rgb',compression='deflate',iccprofile=ICC,
                    extrasamples='unassalpha' if keep_alpha else None,metadata=None,extratags=tags,
                    software='DCU5 Data Transfer v3')
                notes.append('Common camera fields stored in TIFF; standard photographic EXIF is retained in the JSON record.')
            elif fmt=='png':f.write(encode_png(result,photo.exif))
            else:
                im=Image.fromarray(result)
                if fmt=='jpeg':im.save(f,format='JPEG',quality=jpeg_quality,subsampling=0,icc_profile=ICC,exif=photo.exif)
                elif fmt=='webp':im.save(f,format='WEBP',lossless=True,exact=True,method=4,icc_profile=ICC,exif=photo.exif)
                else:
                    im.save(f,format='BMP');notes.append('BMP has no embedded ICC or camera EXIF; metadata is in JSON.')
        report={'source':str(source),'output':str(dest),'look':look,'amount':amount,'exposure_ev':exposure,
            'width':int(result.shape[1]),'height':int(result.shape[0]),'input_bits':array.dtype.itemsize*8,
            'output_format':fmt,'output_bits':bits,'alpha_retained':keep_alpha,'jpeg_quality':jpeg_quality if fmt=='jpeg' else None,
            'lossless_encoding':fmt!='jpeg','metadata':photo.metadata,'standard_exif_hex':photo.exif.hex(),
            'source_sha256':file_hash(source),'output_sha256':file_hash(dest),'color_space':'sRGB IEC61966-2.1',
            'notes':notes,'original_engine_equivalence':False}
        with sidecar.open('x',encoding='utf-8') as f:
            sidecar_created=True;json.dump(report,f,ensure_ascii=False,indent=2)
        return report
    except Exception:
        if created:dest.unlink(missing_ok=True)
        if sidecar_created:sidecar.unlink(missing_ok=True)
        raise

def file_hash(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--look',choices=list(LOOKS),default='GOLD')
    p.add_argument('--amount',type=float,default=1);p.add_argument('--exposure',type=float,default=0)
    p.add_argument('--half-size',action='store_true');p.add_argument('--jpeg-quality',type=int,default=98)
    p.add_argument('--bit-depth',type=int,choices=[8,16],default=16)
    a=p.parse_args();print(json.dumps(process(a.input,a.output,a.look,a.amount,a.exposure,a.half_size,
        jpeg_quality=a.jpeg_quality,bit_depth=a.bit_depth),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
