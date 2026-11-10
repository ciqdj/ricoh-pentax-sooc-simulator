"""Generate a standards-typed minimal EXIF test file; never edit input files."""
from pathlib import Path
import struct
from PIL import Image


def make_exif(note, width, height, iso=None, model_name='PENTAX K-3 Mark III'):
    data = bytearray(b'II*\0\x08\0\0\0')

    def ifd(rows):
        rows = sorted(rows)
        start = len(data)
        data.extend(struct.pack('<H', len(rows)) + bytes(12*len(rows)+4))
        for i, (tag, typ, count, value) in enumerate(rows):
            at = start + 2 + 12*i
            struct.pack_into('<HHI', data, at, tag, typ, count)
            if len(value) <= 4:
                data[at+8:at+12] = value.ljust(4, b'\0')
            else:
                if len(data) % 2:
                    data.extend(b'\0')
                struct.pack_into('<I', data, at+8, len(data))
                data.extend(value)
        return start

    make = b'RICOH IMAGING COMPANY, LTD.\0'
    model = model_name.encode('ascii')+b'\0'
    rows = [(271,2,len(make),make),(272,2,len(model),model),
            (274,3,1,struct.pack('<H',1)),
            (282,5,1,struct.pack('<II',72,1)),(283,5,1,struct.pack('<II',72,1)),
            (296,3,1,struct.pack('<H',2)),(531,3,1,struct.pack('<H',1)),
            (34665,4,1,b'\0'*4)]
    root = ifd(rows)
    exif_rows = [(36864,7,4,b'0231'),(37121,7,4,b'\1\2\3\0'),
                (37500,7,len(note),note),(40960,7,4,b'0100'),
                (40961,3,1,struct.pack('<H',1)),
                (40962,4,1,struct.pack('<I',width)),(40963,4,1,struct.pack('<I',height))]
    if iso is not None:
        exif_rows.append((34855,3,1,struct.pack('<H',iso)))
    exif = ifd(exif_rows)
    struct.pack_into('<I',data,root+2+12*7+8,exif)
    return b'Exif\0\0'+data

def replace_jpeg_exif(original, exif):
    assert original[:2] == b'\xff\xd8'
    assert len(exif) + 2 < 65536
    output = bytearray(original[:2])
    output.extend(b'\xff\xe1'+struct.pack('>H',len(exif)+2)+exif)
    pos=2
    while pos < len(original):
        assert original[pos] == 255
        marker=original[pos+1]
        if marker in (0xda,0xd9):
            output.extend(original[pos:]);break
        size=struct.unpack_from('>H',original,pos+2)[0]
        segment=original[pos:pos+2+size]
        if not (marker==0xe1 and segment[4:10]==b'Exif\0\0'):
            output.extend(segment)
        pos+=2+size
    return bytes(output)

import numpy as np
def _tiff_header(rgb,note,iso=100,model_name='PENTAX K-3 Mark III'):
    assert rgb.dtype==np.uint16 and rgb.ndim==3 and rgb.shape[2]==3
    h,w=rgb.shape[:2]
    data=bytearray(make_exif(note,w,h,iso=iso,model_name=model_name)[6:])
    oldroot=struct.unpack_from('<I',data,4)[0]
    n=struct.unpack_from('<H',data,oldroot)[0]
    exif=next(struct.unpack_from('<I',data,oldroot+10+i*12)[0] for i in range(n) if struct.unpack_from('<H',data,oldroot+2+i*12)[0]==34665)
    def short(t,v):return(t,3,1,struct.pack('<H',v))
    def long(t,v):return(t,4,1,struct.pack('<I',v))
    def ascii_(t,v):return(t,2,len(v)+1,v.encode('ascii')+b'\0')
    rows=[long(256,w),long(257,h),(258,3,3,struct.pack('<3H',16,16,16)),
          short(259,1),short(262,2),ascii_(271,'RICOH IMAGING COMPANY, LTD.'),
          ascii_(272,model_name),long(273,0),short(274,1),
          short(277,3),long(278,h),long(279,rgb.nbytes),
          (282,5,1,struct.pack('<II',72,1)),(283,5,1,struct.pack('<II',72,1)),
          short(284,1),short(296,2),long(34665,exif)]
    if len(data)%2:data.extend(b'\0')
    root=len(data);rows.sort();struct.pack_into('<I',data,4,root)
    data.extend(struct.pack('<H',len(rows))+bytes(12*len(rows)+4))
    strip_pos=None
    for i,(tag,typ,count,value) in enumerate(rows):
        at=root+2+i*12;struct.pack_into('<HHI',data,at,tag,typ,count)
        if len(value)<=4:data[at+8:at+12]=value.ljust(4,b'\0')
        else:
            if len(data)%2:data.extend(b'\0')
            struct.pack_into('<I',data,at+8,len(data));data.extend(value)
        if tag==273:strip_pos=at+8
    if len(data)%2:data.extend(b'\0')
    struct.pack_into('<I',data,strip_pos,len(data))
    return bytes(data)


def tiff_rgb(rgb,note,iso=100,model_name='PENTAX K-3 Mark III'):
    return _tiff_header(rgb,note,iso,model_name)+np.asarray(rgb,dtype='<u2').tobytes()


def write_tiff_rgb(path,rgb,note,iso=100,model_name='PENTAX K-3 Mark III'):
    """Write the exact same TIFF bytes without duplicating the entire image."""
    with Path(path).open('wb') as out:
        out.write(_tiff_header(rgb,note,iso,model_name))
        for start in range(0,rgb.shape[0],128):
            out.write(np.ascontiguousarray(rgb[start:start+128],dtype='<u2').tobytes())
