"""User-requested post-filter Lab balance; NOT camera white-balance emulation.

Horizontal: blue (-b*) to amber (+b*). Vertical: magenta (+a*) to
green (-a*). Values are Lab offsets, not the camera's white-balance steps.
The zero setting bypasses computation and preserves every original pixel.
"""
import numpy as np
from PIL import Image,ImageTk
from ui_language import tk, ttk

LIMIT=35


def validate(value):
    if not isinstance(value,(list,tuple)) or len(value)!=2:raise ValueError('色彩平衡需要两个坐标')
    if any(type(x) not in (int,float) or not np.isfinite(x) or abs(x)>LIMIT for x in value):raise ValueError('色彩平衡超出范围')
    return tuple(float(x) for x in value)


def transform(rgb,x,y):
    x,y=validate((x,y))
    if x==0 and y==0:return rgb
    v=np.asarray(rgb,dtype=np.float64)
    linear=np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)
    r,g,b=linear[...,0],linear[...,1],linear[...,2]
    xyz=np.stack(((.4124564*r+.3575761*g+.1804375*b)/.95047,
                  .2126729*r+.7151522*g+.0721750*b,
                  (.0193339*r+.1191920*g+.9503041*b)/1.08883),axis=-1)
    d=6/29;f=np.where(xyz>d**3,np.cbrt(xyz),xyz/(3*d*d)+4/29)
    # Keep L*, add the requested a*/b* offsets after all filter processing.
    fy=f[...,1];fx=f[...,0]-y/500;fz=f[...,2]-x/200
    f=np.stack((fx,fy,fz),axis=-1)
    xyz=np.where(f>d,f**3,3*d*d*(f-4/29))*np.array([.95047,1,1.08883])
    X,Y,Z=xyz[...,0],xyz[...,1],xyz[...,2]
    linear=np.stack((3.2404542*X-1.5371385*Y-.4985314*Z,
                    -.9692660*X+1.8760108*Y+.0415560*Z,
                    .0556434*X-.2040259*Y+1.0572252*Z),axis=-1)
    return np.clip(np.where(linear<=.0031308,linear*12.92,1.055*np.maximum(linear,0)**(1/2.4)-.055),0,1)


# ---------------------------------------------------------------------------
# AUDIT-P23: interactive balance needs ~35 FPS at 480x360, but one
# RGB->Lab->RGB round trip costs ~107 ms because it does three sRGB decodes
# and three sRGB encodes with pow().
#
# The math is separable: while the pointer moves the source pixels do not
# change, so RGB->Lab can be computed once and only Lab->RGB repeated.
# prepare_rgb16()/apply_lab16() expose that split. apply_rgb16() keeps the
# plain one-shot contract and NEVER caches, so no caller can ever see stale
# pixels. All three produce results bit-identical to transform().
# ---------------------------------------------------------------------------
_D = 6/29
_D3 = _D**3
_M1 = np.array([[.4124564,.3575761,.1804375],
                [.2126729,.7151522,.0721750],
                [.0193339,.1191920,.9503041]]) / np.array([.95047,1,1.08883])[:,None]
_M1T = np.ascontiguousarray(_M1.T)
_M2 = np.array([[3.2404542,-1.5371385,-.4985314],
                [-.9692660,1.8760108,.0415560],
                [.0556434,-.2040259,1.0572252]])
_M2T = np.ascontiguousarray(_M2.T)
_SCALE = np.array([.95047,1,1.08883])


def _srgb_to_lab(v):
    """v: float64 sRGB 0..1 -> the Lab-side cbrt vector."""
    linear = np.where(v<=.04045, v/12.92, ((v+.055)/1.055)**2.4)
    xyz = linear @ _M1T
    out = np.empty_like(xyz)
    big = xyz > _D3
    np.cbrt(xyz, out=out, where=big)
    np.divide(xyz, 3*_D*_D, out=out, where=~big)
    out[~big] += 4/29
    return out


def _lab_to_srgb(f, ax, ay):
    """f: float64 Lab-side vector -> sRGB 0..1. ax=A-B, ay=G-M."""
    g = f.copy()
    g[...,0] -= ay/500.0
    g[...,2] -= ax/200.0
    xyz = np.empty_like(g)
    big = g > _D
    np.multiply(g, g, out=xyz)
    xyz *= g
    small = np.subtract(g, 4/29, out=np.empty_like(g))
    small *= 3*_D*_D
    np.copyto(xyz, small, where=~big)
    xyz *= _SCALE
    linear = xyz @ _M2T
    out = np.empty_like(linear)
    nb = linear > .0031308
    np.multiply(linear, 12.92, out=out, where=~nb)
    pos = np.maximum(linear, 0, out=np.empty_like(linear))
    np.power(pos, 1/2.4, out=pos)
    pos *= 1.055
    pos -= .055
    np.copyto(out, pos, where=nb)
    np.clip(out, 0, 1, out=out)
    return out


def prepare_rgb16(rgb):
    """Compute the reusable RGB->Lab half once for an immutable RGB16 input.

    The caller owns the lifetime. It must only reuse the result while the
    exact same pixel content is still valid; editor_pipeline keys this on its
    mixer-stage cache entry, which is never mutated in place.
    """
    if rgb.dtype!=np.uint16 or rgb.ndim!=3 or rgb.shape[-1]!=3:
        raise ValueError('色彩平衡需要 RGB16 输入')
    return _srgb_to_lab(rgb.astype(np.float64)/65535)


def apply_lab16(lab,balance=(0,0),cancelled=None):
    """Lab -> RGB16 for one balance. Bit-identical to transform()."""
    x,y=validate(balance)
    if lab.dtype!=np.float64 or lab.ndim!=3 or lab.shape[-1]!=3:
        raise ValueError('色彩平衡需要预计算的 Lab 输入')
    out=np.empty(lab.shape,np.uint16)
    for row in range(0,len(lab),64):
        if cancelled and cancelled():
            from native_backend import RenderCancelled
            raise RenderCancelled()
        piece=_lab_to_srgb(lab[row:row+64],x,y)
        out[row:row+64]=np.floor(piece*65535+.5).astype(np.uint16)
    return out


def apply_rgb16(rgb,balance=(0,0),cancelled=None):
    """One-shot RGB16 balance. Recomputes everything; holds no state."""
    x,y=validate(balance)
    if x==0 and y==0:return rgb
    if rgb.dtype!=np.uint16 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('色彩平衡需要 RGB16 输入')
    from gpu_compute import balance as gpu_balance
    accelerated=gpu_balance(rgb,x,y,cancelled)
    if accelerated is not None:return accelerated
    lab=_srgb_to_lab(rgb.astype(np.float64)/65535)
    return apply_lab16(lab,(x,y),cancelled)

def apply_image(image,balance=(0,0)):
    x,y=validate(balance)
    if x==0 and y==0:return image
    from gpu_compute import balance as gpu_balance
    accelerated=gpu_balance(np.asarray(image.convert('RGB')),x,y)
    if accelerated is not None:return Image.fromarray(accelerated)
    return Image.fromarray(np.floor(transform(np.array(image.convert('RGB'),dtype=np.float64)/255,x,y)*255+.5).astype(np.uint8))


class BalancePad(ttk.Frame):
    def __init__(self,parent,command):
        super().__init__(parent);self.command=command;self.value=(0,0);self.enabled=True
        self.canvas=tk.Canvas(self,width=250,height=238,bg='#2c2c2e',highlightthickness=0,takefocus=True,cursor='crosshair')
        self.canvas.pack(anchor='center')
        # Colour gradient is generated from the same transform as the photo.
        axis=np.linspace(-LIMIT,LIMIT,71);xx,yy=np.meshgrid(axis,axis[::-1]);pixels=np.empty((71,71,3),np.uint8)
        for row in range(71):
            for col in range(71):pixels[row,col]=np.floor(transform(np.array([.6,.6,.6]),float(xx[row,col]),float(yy[row,col]))*255+.5).astype(np.uint8)
        self.palette=ImageTk.PhotoImage(Image.fromarray(pixels).resize((184,184),Image.Resampling.BILINEAR))
        self.canvas.create_image(125,119,image=self.palette)
        for delta in (-.5,0,.5):
            c=125+delta*184;self.canvas.create_line(c,27,c,211,fill='#55575b' if delta else '#d0d0d5')
            c=119+delta*184;self.canvas.create_line(33,c,217,c,fill='#55575b' if delta else '#d0d0d5')
        for x,y,text in ((125,12,'G 绿'),(125,226,'M 洋红'),(15,119,'B\n蓝'),(234,119,'A\n琥珀')):
            self.canvas.create_text(x,y,text=text,fill='#f2f2f7',font=('Microsoft YaHei UI',9))
        self.cursor=self.canvas.create_oval(119,113,131,125,width=2,outline='white',fill='#1c1c1e')
        self.label=ttk.Label(self,text='A–B  0.0    G–M  0.0',anchor='center');self.label.pack(fill='x')
        self.reset_button=ttk.Button(self,text='复位色彩平衡',command=lambda:self.select(0,0));self.reset_button.pack(fill='x',pady=5)
        self.canvas.bind('<Button-1>',self.click);self.canvas.bind('<B1-Motion>',self.click)
        for key,dx,dy in [('Left',-1,0),('Right',1,0),('Up',0,1),('Down',0,-1)]:
            self.canvas.bind('<'+key+'>',lambda e,a=dx,b=dy:self.select(self.value[0]+a,self.value[1]+b))
        self.canvas.bind('<Home>',lambda e:self.select(0,0))

    def set(self,value):
        self.value=validate(value);x,y=self.value;cx=125+x/LIMIT*92;cy=119-y/LIMIT*92
        self.canvas.coords(self.cursor,cx-6,cy-6,cx+6,cy+6)
        self.label.configure(text=f'A–B  {x:+.1f}    G–M  {y:+.1f}')

    def select(self,x,y):
        if not self.enabled:return
        value=(round(max(-LIMIT,min(LIMIT,x)),1),round(max(-LIMIT,min(LIMIT,y)),1))
        self.set(value);self.command(value)

    def click(self,event):
        self.canvas.focus_set();self.select((event.x-125)/92*LIMIT,(119-event.y)/92*LIMIT)

    def set_enabled(self,enabled):
        self.enabled=enabled;self.reset_button.configure(state='normal' if enabled else 'disabled')
        self.canvas.configure(cursor='crosshair' if enabled else 'arrow')
