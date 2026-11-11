"""Explicit post-filter tonal editing, with exact neutral RGB16 bypass."""
from functools import lru_cache
import math
import numpy as np
from PIL import Image
from ui_language import tk, ttk
from ui_controls import CollapsibleSection,Tooltip

DEFAULT={'regions':(0.,)*5,'points':((0.,0.),(1.,1.))}
NAMES=('黑色','阴影','中间调','高光','白色')


def validate(value=None):
    value=DEFAULT if value is None else value
    if not isinstance(value,dict) or set(value)-{'regions','points'}:raise ValueError('明暗调整格式无效')
    regions=tuple(value.get('regions',DEFAULT['regions']))
    points=tuple(tuple(p) for p in value.get('points',DEFAULT['points']))
    if len(regions)!=5 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>2 for x in regions):raise ValueError('分区明暗范围为 −2 至 +2')
    if not 2<=len(points)<=12 or any(len(p)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in p) for p in points):raise ValueError('曲线点无效')
    if points[0][0]!=0 or points[-1][0]!=1 or any(points[i+1][0]-points[i][0]<1/1024 for i in range(len(points)-1)):raise ValueError('曲线横坐标必须严格递增并保留两端')
    return {'regions':regions,'points':points}


def neutral(value=None):
    value=validate(value)
    return not any(value['regions']) and all(x==y for x,y in value['points'])


def interpolate(points,x):
    """Shape-preserving cubic Hermite interpolation, no overshoot between knots."""
    p=np.asarray(points,np.float64);a=p[:,0];b=p[:,1];h=np.diff(a);d=np.diff(b)/h;m=np.zeros(len(p))
    if len(p)==2:m[:]=d[0]
    else:
        for i in range(1,len(p)-1):
            if d[i-1]*d[i]>0:
                w1=2*h[i]+h[i-1];w2=h[i]+2*h[i-1];m[i]=(w1+w2)/(w1/d[i-1]+w2/d[i])
        for i,j,k in ((0,0,1),(-1,-1,-2)):
            v=((2*h[j]+h[k])*d[j]-h[j]*d[k])/(h[j]+h[k])
            m[i]=0 if v*d[j]<=0 else min(abs(v),3*abs(d[j]))*np.sign(v)
    ix=np.clip(np.searchsorted(a,x,side='right')-1,0,len(p)-2);t=(x-a[ix])/h[ix]
    return (2*t**3-3*t*t+1)*b[ix]+(t**3-2*t*t+t)*h[ix]*m[ix]+(-2*t**3+3*t*t)*b[ix+1]+(t**3-t*t)*h[ix]*m[ix+1]


@lru_cache(maxsize=16)
def table(regions,points,bits):
    maximum=(1<<bits)-1;x=np.arange(maximum+1,dtype=np.float64)/maximum
    # Smooth, local brightness shifts in displayed tone. This is a manual
    # post-filter curve, not sensor exposure recovery or camera processing.
    centers=np.linspace(0,1,5);weights=np.maximum(0,1-np.abs(x[:,None]-centers[None,:])*4)
    delta=weights@np.asarray(regions);v=np.clip(x+delta*.125,0,1)
    v=np.clip(interpolate(points,v),0,1)
    return np.floor(v*maximum+.5).astype(np.uint16 if bits==16 else np.uint8)


def apply_array(rgb,value=None,cancelled=None):
    value=validate(value)
    if neutral(value):return rgb
    if rgb.dtype not in (np.uint8,np.uint16) or rgb.shape[-1]!=3:raise ValueError('曲线需要 RGB8/RGB16 输入')
    lut=table(value['regions'],value['points'],rgb.itemsize*8);out=np.empty_like(rgb)
    for row in range(0,len(rgb),128):
        if cancelled and cancelled():
            from native_backend import RenderCancelled
            raise RenderCancelled()
        out[row:row+128]=lut[rgb[row:row+128]]
    return out


def apply_image(image,value=None,cancelled=None):
    if neutral(value):return image
    return Image.fromarray(apply_array(np.asarray(image.convert('RGB')),value,cancelled))


def histogram(image):
    a=np.asarray(image.convert('RGB'));bins=[np.bincount(a[...,c].ravel(),minlength=256) for c in range(3)]
    luminance=((a[...,0].astype(np.uint32)*54+a[...,1].astype(np.uint32)*183+a[...,2].astype(np.uint32)*19+128)//256).astype(np.uint8)
    return {'rgb':bins,'luma':np.bincount(luminance.ravel(),minlength=256),
            'black':float(np.mean(np.all(a==0,axis=-1))), 'white':float(np.mean(np.any(a==255,axis=-1)))}


class TonePanel(ttk.Frame):
    def __init__(self,parent,command):
        super().__init__(parent);self.command=command;self.value=validate();self.active=None;self.drag_start=None;self.enabled=True
        self.hist_section=CollapsibleSection(self,'直方图',expanded=True);self.hist_section.pack(fill='x')
        panel=self.hist_section.body
        self.histogram_source=tk.StringVar(master=self,value='统计来源：尚未选择照片')
        self.hist=tk.Canvas(panel,width=258,height=112,bg='#161618',highlightthickness=0,cursor='sb_h_double_arrow')
        self.hist.pack(fill='x',pady=(5,2));self.hist.bind('<Configure>',lambda e:self.draw_hist())
        self.hist.bind('<Button-1>',self.hist_down);self.hist.bind('<B1-Motion>',self.hist_drag)
        self.hist.bind('<Double-Button-1>',lambda e:self.reset_regions())
        self.hist.bind('<Motion>',self.hist_hover)
        Tooltip(self.hist,lambda:self.histogram_source.get()+'\n拖动调节明暗，双击复位。')
        self.summary=ttk.Label(panel,text='',style='Hint.TLabel',wraplength=258);self.summary.pack(fill='x')
        self.region_label=ttk.Label(panel,text='',style='Hint.TLabel');self.region_label.pack(fill='x')
        self.hist.bind('<Leave>',lambda e:self.region_label.configure(text=''),add='+')
        self.curve_section=CollapsibleSection(self,'曲线',expanded=False);self.curve_section.pack(fill='x',pady=(8,0))
        panel=self.curve_section.body
        self.curve=tk.Canvas(panel,width=258,height=156,bg='#161618',highlightthickness=0,takefocus=True,cursor='crosshair')
        self.curve.pack(fill='x');self.curve.bind('<Configure>',lambda e:self.draw_curve())
        self.curve.bind('<Button-1>',self.curve_down);self.curve.bind('<B1-Motion>',self.curve_drag)
        self.curve.bind('<Button-3>',self.delete_point);self.curve.bind('<Delete>',self.delete_point)
        Tooltip(self.curve,'RGB 曲线：上拉提亮，下拉压暗。\n点击添加控制点，右键删除。')
        row=ttk.Frame(panel);row.pack(fill='x',pady=4)
        for label,points in [('线性',((0,0),(1,1))),('S 曲线',((0,0),(.25,.19),(.75,.81),(1,1)))]:
            ttk.Button(row,text=label,width=0,command=lambda p=points:self.set_points(p)).pack(side='left',padx=(0,4))
        self.data=None;self.empty_text='等待照片';self.draw_hist();self.draw_curve()

    def set(self,value):
        value=validate(value)
        # A widget's own edit is echoed back by photo-state synchronization.
        # That echo must not release the point held by the mouse. A different
        # state (history/reset/another photo) still resets the selection.
        if value==self.value:return
        self.value=value;self.active=None;self.draw_curve()
        # AUDIT-P34: the histogram data only changes in set_histogram(). Redrawing
        # three 1538-argument polylines plus three percentiles on every drag frame
        # was pure overhead and was the main reason the curve felt heavy.

    def set_enabled(self,enabled):self.enabled=enabled

    def changed(self):
        self.value=validate(self.value);self.draw_curve();self.draw_hist();self.command(self.value)

    def set_histogram(self,data,source='处理后预览',empty_text='等待照片'):
        self.data=data;self.empty_text=empty_text;self.histogram_source.set('统计来源：'+source);self.draw_hist()

    def draw_hist(self):
        c=self.hist;c.delete('all');w=max(258,c.winfo_width());h=112
        if self.data:
            scale=max(1,max(np.percentile(a,99.5) for a in self.data['rgb']))
            for channel,data,color in zip(('r','g','b'),self.data['rgb'],('#a35b63','#518569','#527caa')):
                pts=[0,h-18]
                # AUDIT-P34: half resolution is visually identical at 258 px wide and
                # halves the argument count handed to Tk's create_line.
                step=2 if len(data)>128 else 1
                for i in range(0,len(data),step):pts.extend((i*(w-1)/255,h-18-min(h-22,float(data[i])/scale*(h-22))))
                pts.extend((w,h-18));c.create_line(*pts,fill=color,width=1,tags=('histogram-channel','histogram-'+channel))
            self.summary.configure(text=f'黑场 {self.data["black"]*100:.2f}%    高光 {self.data["white"]*100:.2f}%')
        else:
            c.create_text(w/2,(h-18)/2,text=self.empty_text,fill='#aeb0b8',font=('Microsoft YaHei UI',10),tags='histogram-empty')
            self.summary.configure(text='')
        for i,name in enumerate(NAMES):
            if i:c.create_line(i*w/5,0,i*w/5,h-18,fill='#343438')
            c.create_text((i+.5)*w/5,h-8,text=name,fill='#c4c4ca',font=('Microsoft YaHei UI',8))

    def hist_down(self,e):
        if not self.enabled:return
        index=min(4,max(0,int(e.x/max(258,self.hist.winfo_width())*5)))
        self.drag_start=(index,e.x,self.value['regions'][index]);self.hist_hover(e)

    def hist_hover(self,e):
        index=min(4,max(0,int(e.x/max(258,self.hist.winfo_width())*5)))
        self.region_label.configure(text=f'{NAMES[index]}  {self.value["regions"][index]:+.2f}')

    def hist_drag(self,e):
        if not self.enabled or self.drag_start is None:return
        index,start,value=self.drag_start;values=list(self.value['regions'])
        values[index]=round(max(-2,min(2,value+(e.x-start)/70)),2);self.value=dict(self.value,regions=values)
        self.changed();self.region_label.configure(text=f'{NAMES[index]}  {values[index]:+.2f}')

    def reset_regions(self):
        if self.enabled:self.value=dict(self.value,regions=(0.,)*5);self.changed()

    def coords(self):return max(258,self.curve.winfo_width()),156

    def event_point(self,e):
        w,h=self.coords();return max(0,min(1,(e.x-10)/(w-20))),max(0,min(1,1-(e.y-10)/(h-20)))

    def draw_curve(self):
        c=self.curve;c.delete('all');w,h=self.coords()
        for t in np.linspace(0,1,5):
            c.create_line(10+t*(w-20),10,10+t*(w-20),h-10,fill='#363639')
            c.create_line(10,10+t*(h-20),w-10,10+t*(h-20),fill='#363639')
        c.create_line(10,h-10,w-10,10,fill='#5a5a60',dash=(3,3))
        # AUDIT-P34: 192 samples across a ~250 px canvas is sub-pixel redundancy; 96
        # halves the create_line payload with no visible change on the drawn curve.
        x=np.linspace(0,1,96);y=interpolate(self.value['points'],x);pts=np.column_stack((10+x*(w-20),10+(1-y)*(h-20))).reshape(-1)
        c.create_line(*pts,fill='#e5e5ec',width=2)
        for i,(x,y) in enumerate(self.value['points']):
            x=10+x*(w-20);y=10+(1-y)*(h-20);c.create_oval(x-4,y-4,x+4,y+4,fill='#0a84ff' if i==self.active else '#ededf0',outline='')

    def curve_down(self,e):
        if not self.enabled:return
        self.curve.focus_set();x,y=self.event_point(e);p=list(self.value['points'])
        distance=[(a-x)**2+(b-y)**2 for a,b in p];i=int(np.argmin(distance))
        if distance[i]>.005 and len(p)<12 and .005<x<.995:
            if min(abs(a-x) for a,b in p)<1/1024:return
            p.append((round(x,5),round(y,5)));p.sort();self.active=p.index((round(x,5),round(y,5)))
            self.value=dict(self.value,points=p);self.changed()
        else:self.active=i;self.draw_curve()

    def curve_drag(self,e):
        if not self.enabled or self.active is None:return
        x,y=self.event_point(e);p=list(self.value['points']);i=self.active
        x=0 if i==0 else 1 if i==len(p)-1 else min(p[i+1][0]-.001,max(p[i-1][0]+.001,x))
        p[i]=(round(x,5),round(y,5));self.value=dict(self.value,points=p);self.changed()

    def delete_point(self,e=None):
        if not self.enabled:return
        p=list(self.value['points']);i=self.active
        if e is not None and getattr(e,'num',None)==3:
            x,y=self.event_point(e);i=int(np.argmin([(a-x)**2+(b-y)**2 for a,b in p]))
        if i is not None and 0<i<len(p)-1:
            p.pop(i);self.active=None;self.value=dict(self.value,points=p);self.changed()
        return 'break'

    def set_points(self,points):
        if self.enabled:self.value=dict(self.value,points=points);self.active=None;self.changed()
