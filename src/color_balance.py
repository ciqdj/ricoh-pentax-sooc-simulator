"""User-requested post-filter Lab balance; NOT camera white-balance emulation.

Horizontal: blue (-b*) to amber (+b*). Vertical: magenta (+a*) to
green (-a*). Values are Lab offsets, not the camera's white-balance steps.
The zero setting bypasses computation and preserves every original pixel.
"""
import numpy as np
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import ttk

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


def apply_rgb16(rgb,balance=(0,0),cancelled=None):
    x,y=validate(balance)
    if x==0 and y==0:return rgb
    if rgb.dtype!=np.uint16 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('色彩平衡需要 RGB16 输入')
    from gpu_compute import balance as gpu_balance
    accelerated=gpu_balance(rgb,x,y,cancelled)
    if accelerated is not None:return accelerated
    out=np.empty_like(rgb)
    for row in range(0,len(rgb),64):
        if cancelled and cancelled():
            from native_backend import RenderCancelled
            raise RenderCancelled()
        out[row:row+64]=np.floor(transform(rgb[row:row+64].astype(np.float64)/65535,x,y)*65535+.5).astype(np.uint16)
    return out


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
