"""Compact native Tk editing controls; no global input capture or UI threads."""
import math
import tkinter as tk
from tkinter import ttk


class Tooltip:
    """Short on-demand help, outside the editing layout."""
    def __init__(self,widget,text):
        self.widget=widget;self.text=text;self.pending=None;self.window=None
        widget.bind('<Enter>',self.schedule,add='+')
        for event in ('<Leave>','<ButtonPress>','<Destroy>'):
            widget.bind(event,self.hide,add='+')

    def schedule(self,event=None):
        self.hide();self.pending=self.widget.after(650,self.show)

    def show(self):
        self.pending=None
        if not self.widget.winfo_viewable():return
        value=self.text() if callable(self.text) else self.text
        if not value:return
        window=tk.Toplevel(self.widget);window.withdraw();window.overrideredirect(True)
        self.window=window
        tk.Label(window,text=value,bg='#333336',fg='#eeeeef',padx=10,pady=7,
                 justify='left',wraplength=300,font=('Microsoft YaHei UI',9)).pack()
        window.update_idletasks()
        x=min(self.widget.winfo_pointerx()+12,self.widget.winfo_screenwidth()-window.winfo_reqwidth()-8)
        y=min(self.widget.winfo_pointery()+20,self.widget.winfo_screenheight()-window.winfo_reqheight()-8)
        window.geometry(f'+{max(0,x)}+{max(0,y)}')
        if __import__('os').name=='nt':
            import ctypes
            from ctypes import wintypes as W
            user=ctypes.WinDLL('user32')
            user.GetAncestor.argtypes=[W.HWND,W.UINT];user.GetAncestor.restype=W.HWND
            user.GetWindowLongPtrW.argtypes=[W.HWND,ctypes.c_int];user.GetWindowLongPtrW.restype=ctypes.c_ssize_t
            user.SetWindowLongPtrW.argtypes=[W.HWND,ctypes.c_int,ctypes.c_ssize_t]
            hwnd=user.GetAncestor(window.winfo_id(),2)
            user.SetWindowLongPtrW(hwnd,-20,user.GetWindowLongPtrW(hwnd,-20)|0x08000080)
        window.deiconify()

    def hide(self,event=None):
        if self.pending is not None:
            try:self.widget.after_cancel(self.pending)
            except tk.TclError:pass
            self.pending=None
        if self.window is not None:
            try:self.window.destroy()
            except tk.TclError:pass
            self.window=None


class FineSlider(ttk.Frame):
    """Scale-compatible fine control with a locally editable numeric value.

    Commands receive a numeric string, as with ttk.Scale. Programmatic variable
    writes update the display without firing a command. ``set`` fires a command.
    Shift uses one tenth of the normal step, except for integer-only variables.
    ``default`` is the double-click reset value; it can be changed after reading
    a photo's own factory defaults. Wheel events are left to the scroll panel.
    """
    def __init__(self,parent,variable=None,from_=0.,to=100.,command=None,
                 *,resolution=1.,gradient=None,default=None,width=150,**kwargs):
        super().__init__(parent,**kwargs)
        self._from=self._number(from_);self._to=self._number(to)
        self._resolution=self._positive(resolution);self._command=command
        self._gradient=gradient;self._variable=variable if variable is not None else tk.DoubleVar(self,value=from_)
        self._integer=isinstance(self._variable,tk.IntVar);self._trace=None
        self._guard=False;self._hot=False;self._focused=False;self._drag_x=None;self._drag_value=None
        self._last=self._clamp(self._read(self._from));self._default=self._clamp(self._number(default)) if default is not None else self._last
        # The workbench installs its dark ttk styles after constructing child
        # widgets. Canvas/Entry colours do not follow later ttk style changes.
        background='#1c1c1e'
        self.columnconfigure(0,weight=1)
        self.canvas=tk.Canvas(self,width=width,height=24,background=background,bd=0,
                              highlightthickness=0,takefocus=True,cursor='hand2')
        self.canvas.grid(row=0,column=0,sticky='ew')
        self.text=tk.StringVar(self,value=self._format(self._last))
        self.entry=tk.Entry(self,textvariable=self.text,width=7,justify='right',bd=0,
                            highlightthickness=0,bg=background,fg='#e5e5ea',insertbackground='#e5e5ea',
                            disabledbackground=background,disabledforeground='#636366',
                            readonlybackground=background,selectbackground='#0a84ff',selectforeground='white',
                            font=('Segoe UI',9))
        self.entry.grid(row=0,column=1,sticky='e',padx=(7,0))
        self.canvas.bind('<Configure>',lambda event:self._draw())
        self.canvas.bind('<Enter>',lambda event:self._hover(True))
        self.canvas.bind('<Leave>',lambda event:self._hover(False))
        self.canvas.bind('<FocusIn>',lambda event:self._focus(True))
        self.canvas.bind('<FocusOut>',lambda event:self._focus(False))
        self.canvas.bind('<Button-1>',self._press)
        self.canvas.bind('<B1-Motion>',self._drag)
        self.canvas.bind('<ButtonRelease-1>',self._release)
        self.canvas.bind('<Double-Button-1>',self._reset)
        for key in ('Left','Right','Up','Down','Home','End','Prior','Next'):
            self.canvas.bind('<'+key+'>',self._key)
            if key in ('Up','Down','Prior','Next'):self.entry.bind('<'+key+'>',self._key)
        self.entry.bind('<Return>',self._commit)
        self.entry.bind('<FocusOut>',self._commit)
        self.entry.bind('<Escape>',self._revert)
        self.bind('<Destroy>',self._destroyed,add='+')
        self._trace=self._variable.trace_add('write',self._variable_changed)
        self._variable_changed();self._draw()

    @staticmethod
    def _number(value):
        value=float(value)
        if not math.isfinite(value):raise ValueError('Slider value must be finite')
        return value

    @classmethod
    def _positive(cls,value):
        value=cls._number(value)
        if value<=0:raise ValueError('Slider resolution must be positive')
        return value

    def _read(self,fallback):
        try:return self._number(self._variable.get())
        except (ValueError,TypeError,tk.TclError):return fallback

    def _clamp(self,value):
        return min(max(self._from,self._to),max(min(self._from,self._to),value))

    @staticmethod
    def _format(value):
        # Sufficient precision for exposure and fine colour adjustments without
        # displaying binary floating-point tails or a long padded decimal.
        return f'{0. if abs(value)<1e-12 else value:.10f}'.rstrip('0').rstrip('.')

    def _step(self,shift=False):
        step=self._resolution*(.1 if shift else 1.)
        return max(1.,round(step)) if self._integer else step

    def _interactive(self):return not self.instate(('disabled',)) and not self.instate(('readonly',))

    def _variable_changed(self,*args):
        if self._guard:return
        value=self._clamp(self._read(self._last))
        if self._integer:value=round(value)
        self._last=value;self._guard=True
        try:
            # Keep the bound model and visible editor consistent even after an
            # external out-of-range or non-finite value is written.
            self._variable.set(int(value) if self._integer else value)
            self.text.set(self._format(value))
        finally:self._guard=False
        self._draw()

    def _write(self,value,step=None,invoke=True):
        value=self._clamp(self._number(value))
        if step:
            value=self._clamp(self._from+round((value-self._from)/step)*step)
        value=round(value,10)
        if self._integer:value=round(value)
        previous=self._last;self._variable.set(int(value) if self._integer else value)
        if invoke and previous!=self._last and self._command is not None:
            self._command(self._format(self._last))

    def get(self):return self._last
    def set(self,value):
        if self._interactive():self._write(value)

    def _geometry(self):
        width=self.canvas.winfo_width()
        if width<20:width=int(float(self.canvas.cget('width')))
        return 8.,max(9.,width-8.),12.

    def _position(self):
        left,right,_=self._geometry()
        fraction=(self._last-self._from)/(self._to-self._from) if self._to!=self._from else 0.
        return left+fraction*(right-left)

    def _point_value(self,x):
        left,right,_=self._geometry()
        return self._clamp(self._from+(x-left)/max(1.,right-left)*(self._to-self._from))

    def _hover(self,value):self._hot=value;self._draw()
    def _focus(self,value):self._focused=value;self._draw()

    def _press(self,event):
        if not self._interactive():return 'break'
        self.canvas.focus_set()
        if abs(event.x-self._position())>8:
            self._write(self._point_value(event.x),self._step(bool(event.state&1)))
        self._drag_x=event.x;self._drag_value=self._last;self._draw()
        return 'break'

    def _drag(self,event):
        if not self._interactive() or self._drag_x is None:return 'break'
        left,right,_=self._geometry();shift=bool(event.state&1)
        self._drag_value=self._clamp(self._drag_value+(event.x-self._drag_x)/max(1.,right-left)
                                    *(self._to-self._from)*(.1 if shift else 1.))
        self._drag_x=event.x
        self._write(self._drag_value,self._step(shift));return 'break'

    def _release(self,event=None):self._drag_x=None;self._drag_value=None;self._draw();return 'break'

    def _reset(self,event=None):
        if self._interactive():self._release();self._write(self._default)
        return 'break'

    def _key(self,event):
        if not self._interactive():return 'break'
        if event.keysym=='Home':value=self._from
        elif event.keysym=='End':value=self._to
        else:
            direction=1 if event.keysym in ('Right','Up','Prior') else -1
            value=self._last+direction*self._step(bool(event.state&1))*(10 if event.keysym in ('Prior','Next') else 1)
        self._write(value);return 'break'

    def _commit(self,event=None):
        if not self._interactive():return 'break'
        try:self._write(self.text.get().strip())
        except (ValueError,TypeError):self.text.set(self._format(self._last))
        else:self.text.set(self._format(self._last))
        return 'break' if event is None or getattr(event,'keysym','')=='Return' else None

    def _revert(self,event=None):self.text.set(self._format(self._last));return 'break'

    def _draw(self):
        if not hasattr(self,'canvas') or not self.canvas.winfo_exists():return
        c=self.canvas;c.delete('all');left,right,y=self._geometry();disabled=not self._interactive()
        c.create_line(left,y,right,y,width=2,capstyle='round',fill='#39393c' if disabled else '#626268',tags='track')
        if self._gradient and not disabled:
            colors=[tuple(channel/257 for channel in self.winfo_rgb(color)) for color in self._gradient]
            if len(colors)>1:
                count=min(64,max(2,int(right-left)))
                for index in range(count):
                    position=(index+.5)/count*(len(colors)-1);at=min(len(colors)-2,int(position));mix=position-at
                    rgb=tuple(round(a*(1-mix)+b*mix) for a,b in zip(colors[at],colors[at+1]))
                    c.create_line(left+index/count*(right-left),y,left+(index+1)/count*(right-left),y,
                                  width=2,fill='#%02x%02x%02x'%rgb,tags='gradient')
        elif not disabled:
            c.create_line(left,y,self._position(),y,width=2,capstyle='round',fill='#0a84ff',tags='active-track')
        if min(self._from,self._to)<0<max(self._from,self._to):
            zero=left+(0-self._from)/(self._to-self._from)*(right-left)
            c.create_line(zero,y-3,zero,y+3,fill='#6b6b70' if not disabled else '#444447',tags='zero-tick')
        x=self._position();radius=4.5 if self._hot or self._drag_x is not None else 4.
        if self._focused and not disabled:c.create_oval(x-7,y-7,x+7,y+7,outline='#0a84ff',width=1,tags='focus-ring')
        c.create_oval(x-radius,y-radius,x+radius,y+radius,outline='',
                      fill='#636366' if disabled else '#ffffff' if self._hot else '#d1d1d6',tags='thumb')

    def _sync_state(self):
        disabled=self.instate(('disabled',));readonly=self.instate(('readonly',))
        self.entry.configure(state='disabled' if disabled else 'readonly' if readonly else 'normal')
        self.canvas.configure(cursor='arrow' if disabled or readonly else 'hand2')
        if disabled or readonly:self._drag_x=None;self._drag_value=None
        self._draw()

    def state(self,statespec=None):
        result=super().state(statespec)
        if statespec is not None and hasattr(self,'entry'):self._sync_state()
        return result

    def cget(self,key):
        key='from' if key=='from_' else key
        values={'from':self._from,'to':self._to,'resolution':self._resolution,
                'command':self._command,'variable':self._variable,'gradient':self._gradient,'default':self._default}
        if key=='state':return 'disabled' if self.instate(('disabled',)) else 'readonly' if self.instate(('readonly',)) else 'normal'
        return values[key] if key in values else super().cget(key)

    __getitem__=cget

    def configure(self,cnf=None,**kwargs):
        if isinstance(cnf,str):
            if cnf in ('from','from_','to','resolution','command','variable','gradient','default','state'):
                return (cnf,cnf,cnf,None,self.cget(cnf))
            return super().configure(cnf)
        options=dict(cnf or {});options.update(kwargs)
        if not options:
            result=super().configure()
            for key in ('from','to','resolution','command','variable','gradient','default','state'):
                result[key]=(key,key,key,None,self.cget(key))
            return result
        if 'variable' in options:
            self._variable.trace_remove('write',self._trace);self._variable=options.pop('variable')
            self._integer=isinstance(self._variable,tk.IntVar)
            self._trace=self._variable.trace_add('write',self._variable_changed)
        for name,attribute in (('from_','_from'),('from','_from'),('to','_to'),('default','_default')):
            if name in options:setattr(self,attribute,self._number(options.pop(name)))
        if 'resolution' in options:self._resolution=self._positive(options.pop('resolution'))
        if 'command' in options:self._command=options.pop('command')
        if 'gradient' in options:self._gradient=options.pop('gradient')
        if 'state' in options:
            value=options.pop('state')
            if value not in ('normal','disabled','readonly'):raise ValueError('Unsupported slider state')
            self.state(('!disabled','!readonly') if value=='normal' else (value,'!readonly' if value=='disabled' else '!disabled'))
        if options:super().configure(**options)
        self._default=self._clamp(self._default);self._variable_changed()

    config=configure

    def _destroyed(self,event):
        if event.widget is not self:return
        if self._trace is not None:
            try:self._variable.trace_remove('write',self._trace)
            except tk.TclError:pass
            self._trace=None
        # Destroying Tcl widgets does not break the Python parent/child cycles.
        # Release their Tk variables now, on the GUI thread, so a later image
        # worker's cyclic collection cannot run Variable.__del__ off-thread.
        self.text=None;self._variable=None;self._command=None


class CollapsibleSection(ttk.Frame):
    """A compact disclosure header that hides, rather than destroys, its body."""
    def __init__(self,parent,title='',expanded=True,command=None,**kwargs):
        title=kwargs.pop('text',title)
        super().__init__(parent,**kwargs)
        self.expanded=bool(expanded);self.command=command
        self.header=ttk.Frame(self,takefocus=True);self.header.pack(fill='x')
        background='#1c1c1e'
        self.arrow=tk.Canvas(self.header,width=15,height=26,bg=background,bd=0,highlightthickness=0)
        self.arrow.pack(side='left')
        self.title_label=ttk.Label(self.header,text=title,font=('Segoe UI',10,'bold'),cursor='hand2')
        self.title_label.pack(side='left',fill='x',expand=True,padx=(2,0))
        self.body=ttk.Frame(self)
        for widget in (self.header,self.arrow,self.title_label):
            widget.bind('<Button-1>',self.toggle)
            widget.bind('<Enter>',lambda event:self._draw_arrow(True))
            widget.bind('<Leave>',lambda event:self._draw_arrow(False))
        self.header.bind('<space>',self.toggle);self.header.bind('<Return>',self.toggle)
        if self.expanded:self.body.pack(fill='x',pady=(3,5))
        self._draw_arrow()

    def _draw_arrow(self,hot=False):
        self.arrow.delete('all')
        points=(3,10,7,14,11,10) if self.expanded else (5,8,9,12,5,16)
        self.arrow.create_line(*points,width=1.5,fill='#f2f2f7' if hot else '#a1a1a6',
                               capstyle='round',joinstyle='round',tags='disclosure')

    def set_expanded(self,value):
        value=bool(value)
        if value==self.expanded:return
        self.expanded=value
        if value:self.body.pack(fill='x',pady=(3,5))
        else:self.body.pack_forget()
        self._draw_arrow()
        if self.command is not None:self.command(value)

    def toggle(self,event=None):self.set_expanded(not self.expanded);return 'break'
    def set_title(self,title):self.title_label.configure(text=title)

    def adopt(self,widget,**placement):
        """Pack an existing sibling above this newer geometry container.

        Tk's ``pack -in`` changes geometry, not the actual window parent or
        stacking order. A newly-created disclosure body otherwise covers old
        controls even though those controls are mapped and have valid sizes.
        """
        placement.pop('in',None);placement.pop('in_',None)
        widget.pack(in_=self.body,**placement)
        widget.lift()


def fold_existing(widget,title=None,expanded=True):
    """Wrap a packed frame while retaining every existing widget reference.

    Tk permits ``pack(in_=...)`` into a descendant of a widget's real parent.
    The wrapper has that same real parent, making its body a valid container.
    No widget is re-created or re-parented. The caller's sibling pack order and
    outer padding are carried by the wrapper; only its body gets hidden.
    """
    previous=getattr(widget,'_fold_section',None)
    if previous is not None:
        if title is not None:previous.set_title(title)
        return previous
    if widget.winfo_manager()!='pack':raise ValueError('fold_existing requires a packed widget')
    placement=dict(widget.pack_info())
    if title is None:
        try:title=str(widget.cget('text'))
        except tk.TclError:title=''
    section=CollapsibleSection(widget.master,title,expanded=expanded)
    section.pack(**placement,before=widget)
    widget.pack_forget()
    if isinstance(widget,ttk.LabelFrame):
        style=ttk.Style(widget);style.layout('Folded.TLabelframe',[])
        widget.configure(text='',style='Folded.TLabelframe')
    section.adopt(widget,fill='both',expand=True)
    section.content=widget;widget._fold_section=section
    return section
