"""Photo processing with explicit format, bit depth, metadata and alpha handling."""
from pathlib import Path
from dataclasses import dataclass
import argparse, hashlib, io, json, math, struct, zlib
import numpy as np
from PIL import Image, ImageCms
import rawpy, tifffile, imagecodecs
from color_math import srgb_encode

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
    """Read colour and geometry only; do not retain personal EXIF for export."""
    info={}; output=Image.Exif(); profile=None; orientation=1
    try:
        with Image.open(path) as im:
            profile=im.info.get('icc_profile'); src=im.getexif()
            orientation=int(src.get(274,1))
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
        decoder=rawpy
        if ext=='.raf':
            # LibRaw 0.22.3 fixes nondeterministic X-Trans tile interpolation.
            # Keep the regular decoder's optional DNG codecs for other formats.
            import rawpy_f
            if rawpy_f.libraw_version!=(0,22,3):
                raise RuntimeError('富士 RAW 解码组件版本不正确，请重新下载完整程序。')
            decoder=rawpy_f
        with decoder.imread(str(path)) as raw:
            array=raw.postprocess(use_camera_wb=True,use_auto_wb=False,output_color=decoder.ColorSpace.sRGB,
                output_bps=16,gamma=(1,1),no_auto_bright=True,half_size=half_size)
        linear=True;profile=None;orientation=1
        notes.append('RAW developed by LibRaw with camera white balance, not DCU5 demosaicing. RAW originals are never rewritten.')
        if ext=='.raf':notes.append('RAF decoder: LibRaw 0.22.3 with original X-Trans fixes and OpenMP.')
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
        elif array.shape[-1]==2 and page.photometric==1:
            array=np.concatenate([np.repeat(array[...,:1],3,axis=2),array[...,1:2]],axis=2)
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
