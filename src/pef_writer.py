"""Write a non-Pentax Bayer frame into a real PEF container.

AUDIT-P42: the original engine only develops Pentax PEF. Feeding it RGB
through a forged TIFF skips the RAW half of its pipeline, so every other
brand lost the tone curve the camera look depends on. PEF turns out to be
TIFF plus a private lossless codec: DPCM against the same-colour neighbour
two pixels away, coded with a fixed canonical Huffman table stored in
MakerNote tag 0x220. Re-encoding that stream lets any Bayer frame reach the
same develop path a Pentax original takes.
"""
from pathlib import Path
import ctypes as C
import struct

import numpy as np

NATIVE = Path(__file__).resolve().parent/'native'
SIZES = {1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8}
BLACK = 64
WHITE = 16383


class CodeTable(C.Structure):
    _fields_ = [('width', C.c_uint8*15), ('prefix', C.c_uint16*15)]


def _note_fields(note):
    if note[:10] != b'PENTAX \x00II':
        raise ValueError('Unexpected MakerNote layout')
    rows = {}
    for i in range(struct.unpack_from('<H', note, 10)[0]):
        at = 12+i*12
        tag, kind, count, value = struct.unpack_from('<HHII', note, at)
        n = SIZES.get(kind,1)*count
        offset = at+8 if n <= 4 else value
        rows[tag] = (at, kind, count, note[offset:offset+n])
    return rows


def _ifd_entries(data, offset):
    count = struct.unpack_from('<H', data, offset)[0]
    return {struct.unpack_from('<H', data, offset+2+12*i)[0]: offset+2+12*i
            for i in range(count)}


def _maker_note(data):
    first = _ifd_entries(data, struct.unpack_from('<I', data, 4)[0])
    exif = _ifd_entries(data, struct.unpack_from('<I', data, first[34665]+8)[0])
    entry = exif[37500]
    count = struct.unpack_from('<I', data, entry+4)[0]
    offset = struct.unpack_from('<I', data, entry+8)[0]
    return entry, data[offset:offset+count]


def _huffman(note):
    tag = _note_fields(note).get(0x220)
    if tag is None:
        raise ValueError('PEF metadata carries no Huffman table')
    blob = tag[3]
    depth = (struct.unpack_from('<H', blob, 0)[0] + 12) & 15
    first = [struct.unpack_from('<H', blob, 14+2*j)[0] for j in range(depth)]
    width = list(blob[14+depth*2:14+depth*2+depth])
    table = [0]*4097
    for index in range(depth):
        i = first[index]
        end = (first[index] + (4096 >> width[index]) - 1) & 4095
        while i <= end:
            i += 1
            table[i] = (width[index] << 8) | index
    table[0] = 12
    codes = {}
    for index in range(4096):
        value = table[index+1]
        bits, length = value >> 8, value & 0xff
        if bits and (length not in codes or bits < codes[length][0]):
            codes[length] = (bits, index >> (12-bits))
    return codes


class PefTemplate:
    """A real PEF kept as the container skeleton for synthetic frames."""

    def __init__(self, path):
        data = Path(path).read_bytes()
        if data[:4] != b'II*\x00':
            raise ValueError('PEF template must be a little-endian TIFF')
        entries = _ifd_entries(data, struct.unpack_from('<I', data, 4)[0])
        self.data = data
        self.entries = entries
        self.strip = struct.unpack_from('<I', data, entries[273]+8)[0]
        self.rows = struct.unpack_from('<I', data, entries[257]+8)[0]
        self.columns = struct.unpack_from('<I', data, entries[256]+8)[0]
        self.codes = _huffman(_maker_note(data)[1])

    def encode(self, bayer, dll):
        rows, columns = bayer.shape
        if (rows, columns) != (self.rows, self.columns):
            raise ValueError(f'PEF template is {self.columns}x{self.rows}; '
                             f'frame is {columns}x{rows}')
        table = CodeTable()
        for length in range(15):
            bits, prefix = self.codes.get(length, (99, 0))
            table.width[length] = bits
            table.prefix[length] = prefix
        capacity = rows*columns*2 + 1024
        out = np.zeros(capacity, dtype=np.uint8)
        payload = np.ascontiguousarray(bayer, dtype=np.uint16)
        written = dll.pef_encode(
            payload.ctypes.data_as(C.POINTER(C.c_uint16)), rows, columns,
            C.byref(table), out.ctypes.data_as(C.POINTER(C.c_uint8)), capacity)
        if written > capacity:
            raise ValueError('PEF encoder rejected the frame')
        head = bytearray(self.data[:self.strip])
        struct.pack_into('<I', head, self.entries[279]+8, written)
        return bytes(head) + out[:written].tobytes()


_LIBRARY = None


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = C.CDLL(str(NATIVE/'pefenc.dll'))
        _LIBRARY.pef_encode.argtypes = [
            C.POINTER(C.c_uint16), C.c_int, C.c_int,
            C.POINTER(CodeTable), C.POINTER(C.c_uint8), C.c_size_t]
        _LIBRARY.pef_encode.restype = C.c_size_t
    return _LIBRARY


def bayer_from_raw(path):
    """Return (bayer, black, white, wb) at the frame's native resolution."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        sizes = raw.sizes
        frame = np.zeros((sizes.raw_height, sizes.raw_width), dtype=np.uint16)
        visible = raw.raw_image_visible
        frame[sizes.top_margin:sizes.top_margin+visible.shape[0],
              sizes.left_margin:sizes.left_margin+visible.shape[1]] = visible
        black = raw.black_level_per_channel[0]
        white = raw.white_level
        gains = np.array(raw.camera_whitebalance[:3], dtype=np.float64)
    gains = gains/gains[1]
    return frame, black, white, gains


def fit_to(bayer, rows, columns, fill):
    """Place a frame inside the template raster without resampling.

    The engine derives its geometry from the maker-model id, not from the TIFF
    size tags, so a template of one sensor size is reused for every camera and
    the frame is centred in it. Only the covered area carries data.
    """
    canvas = np.full((rows, columns), fill, dtype=np.uint16)
    height, width = bayer.shape
    if height > rows or width > columns:
        top = max((height-rows)//2, 0)
        left = max((width-columns)//2, 0)
        canvas[:] = bayer[top:top+rows, left:left+columns]
    else:
        top = (rows-height)//2
        left = (columns-width)//2
        canvas[top:top+height, left:left+width] = bayer
    return canvas


def normalise(bayer, black, white, source_wb, target_wb, red=0.980, blue=1.085):
    """Map a foreign sensor frame onto the scale and balance Pentax expects.

    The engine assumes the PEF maker's own black/white points and applies its
    own white balance on top, so the frame is rescaled into that range and the
    balance is divided out first. red and blue trim the residual cast the
    engine's colour matrix leaves behind.
    """
    frame = np.maximum(bayer.astype(np.float64)-black, 0.0)
    frame = frame*((WHITE-BLACK)/max(white-black, 1)) + BLACK
    frame[0::2, 0::2] *= (source_wb[0]/target_wb[0])*red
    frame[1::2, 1::2] *= (source_wb[2]/target_wb[2])*blue
    peak = frame.max()
    if peak > WHITE:
        frame *= WHITE/peak
    return np.clip(frame, BLACK, WHITE).astype(np.uint16)
