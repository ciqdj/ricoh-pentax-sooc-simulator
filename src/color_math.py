"""Standard sRGB encoding for decoded input; no creative colour transform."""
import numpy as np

def srgb_decode(x):
    x=np.asarray(x,dtype=np.float64)
    return np.where(x<=.04045,x/12.92,((np.maximum(x,0)+.055)/1.055)**2.4)

def srgb_encode(x):
    x=np.asarray(x,dtype=np.float64)
    return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,0)**(1/2.4)-.055)
