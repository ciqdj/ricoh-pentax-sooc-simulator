"""CUBE and Adobe RGB-table XMP serialization.

Adobe serialization based on RAWmakase (MIT), fixed commit recorded in SOURCES.md.
The CUBE grid is [blue, green, red, channel]; the Adobe grid is [red, green, blue].
"""
from pathlib import Path
import hashlib, struct, zlib
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape, quoteattr
import numpy as np

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?`'|()[]{}@%$#"
CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"

def grid(n):
    b,g,r = np.meshgrid(*([np.linspace(0,1,n)]*3), indexing='ij')
    return np.stack((r,g,b), axis=-1)

def write_cube(path, table, title, comment):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    n = table.shape[0]
    assert table.shape == (n,n,n,3) and np.isfinite(table).all()
    with path.open('w',encoding='ascii',newline='\n') as f:
        f.write('# '+comment.replace('\n',' ')+'\n')
        f.write('TITLE "'+title.replace('"','')+'"\n')
        f.write(f'LUT_3D_SIZE {n}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n')
        np.savetxt(f,table.reshape(-1,3),fmt='%.9f')

def read_cube(path):
    n=None; values=[]
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line=line.partition('#')[0].strip()
        if not line: continue
        parts=line.split()
        if parts[0]=='LUT_3D_SIZE': n=int(parts[1])
        elif parts[0]=='DOMAIN_MIN' and list(map(float,parts[1:])) != [0,0,0]: raise ValueError('Domain must be 0..1')
        elif parts[0]=='DOMAIN_MAX' and list(map(float,parts[1:])) != [1,1,1]: raise ValueError('Domain must be 0..1')
        elif parts[0] in ('TITLE','DOMAIN_MIN','DOMAIN_MAX'): continue
        else: values.append([float(x) for x in parts])
    a=np.asarray(values)
    if n is None or a.shape!=(n**3,3) or not np.isfinite(a).all(): raise ValueError('Invalid CUBE data')
    return a.reshape(n,n,n,3)

def tetrahedral(table, rgb):
    """Six-tetrahedra interpolation, with a separately validated FFmpeg oracle."""
    shape=np.shape(rgb); p=np.clip(np.asarray(rgb,dtype=np.float64).reshape(-1,3),0,1)*(table.shape[0]-1)
    lo=np.minimum(np.floor(p).astype(np.int32),table.shape[0]-2); f=p-lo
    order=np.argsort(-f,axis=1,kind='stable'); sf=np.take_along_axis(f,order,axis=1)
    def node(v): return table[v[:,2],v[:,1],v[:,0]]
    a=lo.copy(); c0=node(a); rows=np.arange(len(a))
    a[rows,order[:,0]]+=1; c1=node(a)
    a[rows,order[:,1]]+=1; c2=node(a)
    a[rows,order[:,2]]+=1; c3=node(a)
    out=(1-sf[:,0,None])*c0+(sf[:,0,None]-sf[:,1,None])*c1+(sf[:,1,None]-sf[:,2,None])*c2+sf[:,2,None]*c3
    return out.reshape(shape)

def adobe_compress(raw):
    stream=struct.pack('<I',len(raw))+zlib.compress(raw)
    chars=[]
    for pos in range(0,len(stream),4):
        part=stream[pos:pos+4]; word=int.from_bytes(part,'little')
        for _ in range(len(part)+1): chars.append(ALPHABET[word%85]); word//=85
    return ''.join(chars)

def adobe_expand(text):
    text=''.join(text.split()); stream=bytearray()
    if len(text)%5==1: raise ValueError('Truncated Adobe base85')
    for pos in range(0,len(text),5):
        part=text[pos:pos+5]; value=sum(ALPHABET.index(c)*85**j for j,c in enumerate(part))
        stream.extend(value.to_bytes(4,'little')[:len(part)-1])
    expected=struct.unpack('<I',stream[:4])[0]; raw=zlib.decompress(stream[4:])
    if len(raw)!=expected: raise ValueError('Wrong decompressed length')
    return raw

def rgb_table_bytes(table):
    n=table.shape[0]
    if not 2<=n<=64: raise ValueError('Adobe RGB tables accept 2..64 grid points')
    samples=np.floor(np.clip(table.transpose(2,1,0,3),0,1)*65535+0.5).astype(np.int64)
    identity=np.floor(grid(n).transpose(2,1,0,3)*65535+0.5).astype(np.int64)
    delta=((samples-identity)&65535).astype('<u2')
    # type 1, version 1, 3 dimensions; sRGB primaries 0, sRGB encoding 1,
    # clipped gamut, amount range 0..2.
    return struct.pack('<4I',1,1,3,n)+delta.tobytes()+struct.pack('<3I2d',0,1,0,0.0,2.0)

def write_xmp(path,table,name,description,group='DCU5 数据移植 · 实验版'):
    raw=rgb_table_bytes(table); digest=hashlib.md5(raw).hexdigest().upper()
    attrs={'PresetType':'Look','Cluster':'','UUID':digest,'SupportsAmount':'True','SupportsColor':'True',
           'SupportsMonochrome':'True','SupportsHighDynamicRange':'False','SupportsNormalDynamicRange':'True',
           'SupportsSceneReferred':'True','SupportsOutputReferred':'True','CameraModelRestriction':'',
           'Version':'10.3','ProcessVersion':'6.7','ConvertToGrayscale':'False','RGBTable':digest,
           'RGBTableAmount':'1.0','Table_'+digest:adobe_compress(raw)}
    xml='<x:xmpmeta xmlns:x="adobe:ns:meta/">\n <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
    xml+='  <rdf:Description rdf:about="" xmlns:crs="'+CRS+'"\n'
    xml+='\n'.join('   crs:'+k+'='+quoteattr(v) for k,v in attrs.items())+'>\n'
    for k,v in [('Name',name),('Group',group),('Description',description)]:
        xml+=f'   <crs:{k}><rdf:Alt><rdf:li xml:lang="x-default">{escape(v)}</rdf:li></rdf:Alt></crs:{k}>\n'
    xml+='  </rdf:Description>\n </rdf:RDF>\n</x:xmpmeta>\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(xml,encoding='utf-8')
    return digest

def read_xmp(path):
    root=ET.parse(path).getroot(); desc=root.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
    digest=desc.attrib['{'+CRS+'}RGBTable']; raw=adobe_expand(desc.attrib['{'+CRS+'}Table_'+digest])
    if hashlib.md5(raw).hexdigest().upper()!=digest: raise ValueError('Table fingerprint mismatch')
    kind,version,dimensions,n=struct.unpack_from('<4I',raw)
    if (kind,version,dimensions)!=(1,1,3) or not 2<=n<=64 or len(raw)!=16+n**3*6+28: raise ValueError('Invalid RGB table')
    tail=struct.unpack_from('<3I2d',raw,16+n**3*6)
    if tail!=(0,1,0,0.,2.): raise ValueError('Unexpected RGB table colour space')
    delta=np.frombuffer(raw,dtype='<u2',count=n**3*3,offset=16).reshape(n,n,n,3).astype(np.int64)
    identity=np.floor(grid(n).transpose(2,1,0,3)*65535+0.5).astype(np.int64)
    table=((delta+identity)&65535)/65535.
    return table.transpose(2,1,0,3)
