"""RAF-only bridge to official LibRaw 0.22.3, compiled with MSVC OpenMP.

Only the rawpy extension's DLL import basename is changed. A separate DLL
preserves the regular decoder's optional DNG codecs. No image code is altered.
"""
from . import _rawpy

libraw_version = _rawpy.libraw_version
ColorSpace = _rawpy.ColorSpace


def imread(path):
    raw = _rawpy.RawPy()
    try:
        raw.open_file(path)
        raw.set_unpack_params(shot_select=0)
    except BaseException:
        raw.close()
        raise
    return raw
