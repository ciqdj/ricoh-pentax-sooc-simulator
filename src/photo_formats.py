"""Photo processing with explicit format, bit depth, metadata and alpha handling."""
from pathlib import Path
from dataclasses import dataclass
import argparse, hashlib, io, json, math, struct, zlib
import numpy as np
from PIL import Image, ImageCms
from PIL.TiffImagePlugin import IFDRational
import rawpy, tifffile, imagecodecs
from color_math import srgb_encode

# Every extension LibRaw can decode, so no camera is rejected at the file dialog.
# Sony keeps .arw today but older bodies write .srf / .sr2; Hasselblad writes .3fr
# on X/907X and .fff on the H series; Pentax pixel-shift writes .arq; Leica and
# Panasonic share .raw / .rwl. Adding one here reaches the dialog, the folder
# scan and the drop handler because all three filter on PHOTO_EXT.
RAW_EXT={'.arw','.srf','.sr2','.pef','.dng','.nef','.nrw','.cr2','.cr3','.crw',
         '.raf','.rw2','.orf','.srw','.rwl','.raw','.3fr','.fff','.arq','.iiq',
         '.kdc','.dcr','.mos','.mrw','.x3f','.mef','.erf','.ia'}
RASTER_EXT={'.jpg','.jpeg','.jpe','.jfif','.png','.tif','.tiff','.webp','.bmp'}
PHOTO_EXT=RAW_EXT|RASTER_EXT
SRGB=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
ICC=SRGB.tobytes()
FORMAT_CHOICES={'跟随输入格式':'auto','JPEG':'jpeg','PNG':'png','TIFF':'tiff','WebP':'webp','BMP':'bmp'}
FORMAT_EXT={'jpeg':'.jpg','png':'.png','tiff':'.tif','webp':'.webp','bmp':'.bmp'}
CAMERA_TAGS=(271,272,306)
CAPTURE_TAGS=(33434,33437,34855,36867,36868,37386,40961,40962,40963,42036)
RATIONAL_TAGS=(33434,33437,37386)

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
    """Read a safe camera-and-capture whitelist, excluding private EXIF blocks."""
    info={}; output=Image.Exif(); profile=None; orientation=1
    try:
        with Image.open(path) as im:
            profile=im.info.get('icc_profile'); src=im.getexif()
            orientation=int(src.get(274,1))
            for tag in (271,272,306):
                if tag in src:output[tag]=src[tag]
            source_ifd=dict(src.get_ifd(34665)) if 34665 in src else {}
            allowed=(33434,33437,34855,36867,36868,37386,40961,40962,40963,42036)
            capture_tags={tag:source_ifd[tag] for tag in allowed if tag in source_ifd}
            if capture_tags:output[34665]=capture_tags
    except (OSError,ValueError,TypeError,KeyError):
        pass
    if Path(path).suffix.lower() in RAW_EXT:
        raw_exif,raw_orientation=_raw_capture_metadata(path)
        for tag in (271,272,306):
            if tag not in output and tag in raw_exif:output[tag]=raw_exif[tag]
        if orientation==1:orientation=raw_orientation
        raw_capture=capture_ifd(raw_exif)
        if raw_capture:
            capture=capture_ifd(output)
            for tag,value in raw_capture.items():capture.setdefault(tag,value)
            output[34665]=capture
    output=_safe_capture(output)
    info=_metadata_info(output)
    return output,profile,orientation,info


def capture_ifd(exif):
    value=exif.get(34665)
    if isinstance(value,dict):return dict(value)
    if value is None:return {}
    try:return dict(exif.get_ifd(34665))
    except (KeyError,TypeError,ValueError,AttributeError):return {}


def _safe_capture(source):
    """One whitelist and explicit EXIF field types for every output format."""
    from fractions import Fraction
    output=Image.Exif()
    for tag in CAMERA_TAGS:
        value=source.get(tag)
        if value is not None:
            value=str(value).strip('\0 ')
            if value:output[tag]=value
    sub={}
    for tag,value in capture_ifd(source).items():
        if tag not in CAPTURE_TAGS or value is None:continue
        try:
            if tag in RATIONAL_TAGS:
                ratio=(Fraction(int(value[0]),int(value[1])) if isinstance(value,(tuple,list)) and len(value)==2
                       else Fraction(str(float(value))).limit_denominator(1000000))
                if ratio<=0 or ratio.numerator>0xffffffff or ratio.denominator>0xffffffff:continue
                value=IFDRational(ratio.numerator,ratio.denominator)
            elif tag in (36867,36868,42036):
                value=str(value).strip('\0 ')
                if not value:continue
            else:
                value=int(value)
                limit=0xffff if tag in (34855,40961) else 0xffffffff
                if not 0<=value<=limit:continue
            sub[tag]=value
        except (ValueError,TypeError,OverflowError,ZeroDivisionError):continue
    if sub:output[34665]=sub
    return output


def _embedded_raw_capture(path):
    """Read container metadata only: Canon CMT tables or RAF JPEG APP1 EXIF."""
    output=Image.Exif();suffix=Path(path).suffix.lower()
    if suffix not in ('.cr3','.raf'):return output
    try:
        with Path(path).open('rb') as stream:
            size=stream.seek(0,2);stream.seek(0)
            if suffix=='.raf':
                header=stream.read(96)
                if len(header)<96 or not header.startswith(b'FUJIFILMCCD-RAW '):return output
                offset,length=struct.unpack_from('>II',header,84)
                end=offset+length
                if length<4 or end>size:return output
                stream.seek(offset)
                if stream.read(2)!=b'\xff\xd8':return output
                while stream.tell()+4<=end:
                    if stream.read(1)!=b'\xff':break
                    marker=stream.read(1)
                    while marker==b'\xff':marker=stream.read(1)
                    if marker in (b'\xda',b'\xd9'):break
                    if marker==b'\x01' or b'\xd0'<=marker<=b'\xd8':continue
                    data=stream.read(2)
                    if len(data)!=2:break
                    length=struct.unpack('>H',data)[0]
                    if length<2 or stream.tell()+length-2>end:break
                    if marker==b'\xe1':
                        data=stream.read(length-2)
                        if data.startswith(b'Exif\0\0'):
                            output.load(data);break
                    else:stream.seek(length-2,1)
            else:
                tables={};budget=[10000]
                def boxes(start,end,depth=0):
                    position=start
                    while position+8<=end and budget[0]>0:
                        budget[0]-=1;stream.seek(position)
                        length,kind=struct.unpack('>I4s',stream.read(8));header=8
                        if length==1:
                            if position+16>end:return
                            length=struct.unpack('>Q',stream.read(8))[0];header=16
                        elif length==0:length=end-position
                        if length<header or position+length>end:return
                        payload=position+header
                        if kind in (b'CMT1',b'CMT2') and length-header<=4*1024*1024:
                            exif=Image.Exif();exif.load(stream.read(length-header));tables[kind]=exif
                        elif kind in (b'moov',b'uuid') and depth<8:
                            boxes(payload+(16 if kind==b'uuid' else 0),position+length,depth+1)
                        position+=length
                boxes(0,size)
                output=tables.get(b'CMT1',output)
                if b'CMT2' in tables:
                    capture=capture_ifd(output)
                    for tag in CAPTURE_TAGS:
                        if tag in tables[b'CMT2']:capture.setdefault(tag,tables[b'CMT2'][tag])
                    if capture:output[34665]=capture
    except (OSError,ValueError,TypeError,KeyError,struct.error):pass
    return output


def _metadata_info(exif):
    info={}
    for tag,name in ((271,'camera_make'),(272,'camera_model'),(306,'date_time')):
        value=exif.get(tag)
        if value not in (None,''):info[name]=str(value)
    sub=capture_ifd(exif)
    for tag,name in ((33434,'exposure_time'),(33437,'f_number'),(34855,'iso'),
                     (36867,'date_time_original'),(37386,'focal_length'),(42036,'lens')):
        value=sub.get(tag)
        if value not in (None,''):info[name]=str(value)
    return info


def _raw_capture_metadata(path):
    """Read non-personal capture fields from RAW metadata without decoding pixels."""
    embedded=_embedded_raw_capture(path)
    exif=Image.Exif();orientation=int(embedded.get(274,1))
    make=embedded.get(271,'');model=embedded.get(272,'');captured=embedded.get(306,'')
    try:
        with tifffile.TiffFile(path) as tf:
            page=tf.pages[0]
            for tag,target in ((271,'make'),(272,'model'),(306,'captured'),(274,'orientation')):
                item=page.tags.get(tag)
                if item is None:continue
                value=item.value
                if target=='orientation':orientation=int(value)
                elif target=='make':make=str(value).strip()
                elif target=='model':model=str(value).strip()
                else:captured=str(value).strip()
    except Exception:
        pass
    capture=capture_ifd(embedded)
    try:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            other=raw.other
            for field,tag in (('shutter_speed',33434),('aperture',33437),('focal_length',37386)):
                value=getattr(other,field,None)
                if value is None:continue
                value=float(value)
                if math.isfinite(value) and value>0:
                    from fractions import Fraction
                    ratio=Fraction(str(value)).limit_denominator(1000000)
                    capture.setdefault(tag,IFDRational(ratio.numerator,ratio.denominator))
            iso=getattr(other,'iso_speed',None)
            if iso is not None and math.isfinite(float(iso)) and float(iso)>0:capture.setdefault(34855,int(round(float(iso))))
            timestamp=getattr(other,'timestamp',None)
            if timestamp is not None and hasattr(timestamp,'strftime'):
                capture.setdefault(36867,timestamp.strftime('%Y:%m:%d %H:%M:%S'))
            lens=getattr(getattr(raw,'lens',None),'model','')
            if lens:capture.setdefault(42036,str(lens).strip())
    except Exception:
        pass
    if make:exif[271]=make
    if model:exif[272]=model
    if captured:exif[306]=captured
    if orientation:exif[274]=orientation
    if capture:
        capture.setdefault(40961,1)
        exif[34665]=capture
    return exif,orientation


def write_tiff_exif(path,exif):
    """Attach the safe EXIF whitelist to a single-page TIFF without re-encoding pixels."""
    if not exif:return False
    source_ifd={tag:exif.get(tag) for tag in (271,272,306,274) if exif.get(tag) not in (None,'')}
    sub=capture_ifd(exif)
    allowed=(33434,33437,34855,36867,36868,37386,40961,40962,40963,42036)
    sub={tag:value for tag,value in sub.items() if tag in allowed and value not in (None,'')}
    if not source_ifd and not sub:return False

    def ascii_value(value):
        data=str(value).encode('ascii','replace')+b'\0'
        return 2,len(data),data

    def rational_value(value):
        from fractions import Fraction
        if isinstance(value,(tuple,list)) and len(value)==2:
            ratio=Fraction(int(value[0]),int(value[1]))
        else:ratio=Fraction(str(float(value))).limit_denominator(1000000)
        if ratio.numerator<0 or ratio.denominator<=0 or ratio.numerator>0xffffffff or ratio.denominator>0xffffffff:
            raise ValueError('EXIF rational is out of range')
        return 5,1,struct.pack(endian+'II',ratio.numerator,ratio.denominator)

    def exif_field(tag,value):
        if tag in (33434,33437,37386):return rational_value(value)
        if tag in (36867,36868,42036):return ascii_value(value)
        if tag in (34855,40961):return 3,1,struct.pack(endian+'H',int(value))
        if tag in (40962,40963):return 4,1,struct.pack(endian+'I',int(value))
        raise ValueError('Unsupported EXIF tag')

    try:
        with Path(path).open('r+b') as stream:
            header=stream.read(8)
            if len(header)<8 or header[:2] not in (b'II',b'MM'):return False
            endian='<' if header[:2]==b'II' else '>'
            if struct.unpack(endian+'H',header[2:4])[0]!=42:return False
            first=struct.unpack(endian+'I',header[4:8])[0]
            stream.seek(first);count_data=stream.read(2)
            if len(count_data)!=2:return False
            count=struct.unpack(endian+'H',count_data)[0]
            raw_entries=[stream.read(12) for _ in range(count)]
            next_data=stream.read(4)
            if any(len(entry)!=12 for entry in raw_entries) or len(next_data)!=4:return False
            next_ifd=struct.unpack(endian+'I',next_data)[0]
            if next_ifd:return False
            entries={struct.unpack(endian+'H',entry[:2])[0]:('raw',entry) for entry in raw_entries}
            for tag,value in source_ifd.items():
                if tag==274:
                    field=(3,1,struct.pack(endian+'H',int(value)))
                else:field=ascii_value(value)
                entries[tag]=('new',field)
            sub_entries={tag:exif_field(tag,value) for tag,value in sub.items()}
            if sub_entries:entries[34665]=('pointer',None)
            if len(entries)>65535 or len(sub_entries)>65535:return False

            stream.seek(0,2);file_end=stream.tell();main_offset=(file_end+1)&~1
            main_size=2+12*len(entries)+4
            main_payload_size=sum(len(item[1][2]) for item in entries.values()
                                  if item[0]=='new' and len(item[1][2])>4)
            exif_offset=(main_offset+main_size+main_payload_size+1)&~1 if sub_entries else 0
            exif_size=2+12*len(sub_entries)+4
            if max(main_offset+main_size+main_payload_size,
                   exif_offset+exif_size if sub_entries else 0)>0xffffffff:return False
            payload_start=main_offset+main_size;payload=bytearray();main_data=[]
            for tag,item in sorted(entries.items()):
                if item[0]=='raw':main_data.append((tag,item[1]));continue
                if item[0]=='pointer':
                    field=(4,1,struct.pack(endian+'I',exif_offset))
                else:field=item[1]
                kind,num,value=field
                if len(value)<=4:slot=value.ljust(4,b'\0')
                else:
                    slot=struct.pack(endian+'I',payload_start+len(payload));payload.extend(value)
                main_data.append((tag,struct.pack(endian+'HHI',tag,kind,num)+slot))
            main=bytearray(struct.pack(endian+'H',len(main_data)))
            for _,entry in main_data:main.extend(entry)
            main.extend(struct.pack(endian+'I',0))
            stream.seek(main_offset);stream.write(main);stream.write(payload)
            if sub_entries:
                exif_payload_start=exif_offset+exif_size;exif_payload=bytearray();exif_data=[]
                for tag,field in sorted(sub_entries.items()):
                    kind,num,value=field
                    if len(value)<=4:slot=value.ljust(4,b'\0')
                    else:
                        slot=struct.pack(endian+'I',exif_payload_start+len(exif_payload));exif_payload.extend(value)
                    exif_data.append(struct.pack(endian+'HHI',tag,kind,num)+slot)
                exif_ifd=bytearray(struct.pack(endian+'H',len(exif_data)))
                for entry in exif_data:exif_ifd.extend(entry)
                exif_ifd.extend(struct.pack(endian+'I',0))
                stream.seek(exif_offset);stream.write(exif_ifd);stream.write(exif_payload)
            stream.seek(4);stream.write(struct.pack(endian+'I',main_offset));stream.flush()
        return True
    except (OSError,ValueError,TypeError,OverflowError,struct.error):
        return False

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
            # AUDIT-P41: LibRaw normalises the frame brightness unless told not to, and
            # turning that off left every non-Pentax RAW far darker than the camera's
            # own rendering: a measured CR2 decoded at 16.7% mean and never exceeded
            # 80% of full scale, so Bright and every other look came out underexposed.
            # Pentax PEF bypasses this path entirely, which is why only other brands
            # looked wrong.
            array=raw.postprocess(use_camera_wb=True,use_auto_wb=False,output_color=decoder.ColorSpace.sRGB,
                output_bps=16,gamma=(1,1),no_auto_bright=False,half_size=half_size)
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


def capture_for_output(source,shape):
    """Return safe capture EXIF for the output's actual orientation and size."""
    if isinstance(source,bytes):
        exif=Image.Exif()
        if source:exif.load(source)
    else:
        exif,_,_,_=metadata_from(source)
    exif=_safe_capture(exif)
    info=_metadata_info(exif)
    exif[274]=1
    sub=capture_ifd(exif)
    sub.update({40961:1,40962:int(shape[1]),40963:int(shape[0])})
    exif[34665]=sub
    return exif.tobytes(),info


def save_raster(path,array,fmt,jpeg_quality=98,keep_alpha=False,exif=b''):
    """Write one new photo with ICC and the same safe capture fields on every route."""
    path=Path(path);created=False
    try:
        with path.open('xb') as out:
            created=True
            if fmt=='tiff':
                tifffile.imwrite(out,array,photometric='rgb',compression='deflate',iccprofile=ICC,
                    extrasamples='unassalpha' if keep_alpha else None,metadata=None,software=None)
            elif fmt=='png':out.write(encode_png(array,exif))
            elif fmt=='jpeg':Image.fromarray(array).save(out,'JPEG',quality=jpeg_quality,subsampling=0,
                                                       icc_profile=ICC,exif=exif)
            elif fmt=='webp':Image.fromarray(array).save(out,'WEBP',lossless=True,exact=True,
                                                       icc_profile=ICC,exif=exif)
            elif fmt=='bmp':Image.fromarray(array).save(out,'BMP')
            else:raise ValueError('不支持的输出格式')
        if fmt=='tiff' and exif:
            parsed=Image.Exif();parsed.load(exif)
            if not write_tiff_exif(path,parsed):raise RuntimeError('TIFF 拍摄参数写入失败')
    except BaseException:
        if created:path.unlink(missing_ok=True)
        raise
