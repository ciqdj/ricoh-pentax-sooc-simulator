"""Translate interface text while keeping the app's stored values stable."""
import json
import os
import re
import tempfile
import tkinter as _tk
from functools import lru_cache
from tkinter import filedialog as _files, messagebox as _messages
from tkinter import simpledialog as _prompts, ttk as _ttk
from weakref import WeakSet

from app_paths import CONFIG
from ui_strings import ENGLISH


PREFERENCE = CONFIG / 'language.json'
OPTIONS = {'中文': 'zh', 'English': 'en'}
_widgets = WeakSet()


def _read_language():
    try:
        value = json.loads(PREFERENCE.read_text(encoding='utf-8'))
        language = value.get('language') if isinstance(value, dict) else None
        if language in OPTIONS.values():
            return language
    except (OSError, ValueError):
        pass
    return 'zh'


_language = _read_language()


def current_language():
    return _language


def set_language(language):
    """Save the selection and refresh existing widgets on the Tk thread."""
    global _language
    if language not in OPTIONS.values():
        raise ValueError('Unsupported interface language')
    if language == _language:
        return
    PREFERENCE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=PREFERENCE.parent, prefix='.language-', delete=False
    ) as stream:
        pending = stream.name
        json.dump({'language': language}, stream)
    try:
        os.replace(pending, PREFERENCE)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)
    _language = language
    for widget in list(_widgets):
        try:
            exists = widget.winfo_exists()
        except _tk.TclError:
            exists = False
        if exists:
            widget._refresh_language()


_HAN = re.compile('[\u3400-\u9fff]')
_PARAMETER = re.compile(r'\{\d+\}')
_TEMPLATES = []
for source, english in ENGLISH.items():
    if _PARAMETER.search(source):
        parts = _PARAMETER.split(source)
        # Prefer the more specific templates when several could match.
        pattern = '(.+?)'.join(map(re.escape, parts))
        _TEMPLATES.append((sum(map(len, parts)), re.compile(pattern, re.DOTALL), english))
_TEMPLATES.sort(key=lambda item: item[0], reverse=True)


@lru_cache(maxsize=2048)
def _english(value):
    if not _HAN.search(value):
        return value
    if value in ENGLISH:
        return ENGLISH[value]
    text = value.strip()
    if text != value:
        begin = len(value) - len(value.lstrip())
        return value[:begin] + _english(text) + value[begin + len(text):]
    for _, pattern, translation in _TEMPLATES:
        match = pattern.fullmatch(text)
        if match:
            return translation.format(*(_english(part) for part in match.groups()))
    for separator in ('\n', ' · ', ' / ', ': '):
        if separator in text:
            parts = [_english(part) for part in text.split(separator)]
            if separator == '\n' and len(parts) == 2 and parts[0].casefold() == parts[1].casefold():
                return parts[0]
            return separator.join(parts)
    return value


def tr(text):
    return _english(text) if _language == 'en' and isinstance(text, str) else text


def source_caption(widget):
    """Read the stable source label when the legacy layout identifies a widget."""
    caption = getattr(widget, '_source_caption', None)
    return widget.cget('text') if caption is None else caption


class _TextWidget:
    def __init__(self, master=None, **options):
        self._source_caption = options.pop('text', None)
        self._source_variable = None
        self._display_variable = None
        source = options.pop('textvariable', None)
        super().__init__(master, **options)
        if source is not None:
            self.configure(textvariable=source)
        self._refresh_language()
        _widgets.add(self)
        self.bind('<Destroy>', self._remove_caption_trace, add='+')

    def _remove_caption_trace(self, event=None):
        if self._source_variable is not None:
            self._source_variable.trace_remove('write', self._caption_trace)
            self._source_variable = None

    def _show_variable(self, *_):
        self._display_variable.set(tr(self._source_variable.get()))

    def configure(self, cnf=None, **options):
        if isinstance(cnf, dict):
            options = {**cnf, **options}
            cnf = None
        if 'text' in options:
            self._source_caption = options['text']
            options['text'] = tr(self._source_caption)
        if 'textvariable' in options:
            self._remove_caption_trace()
            source = options['textvariable']
            if isinstance(source, _tk.Variable):
                self._source_variable = source
                self._display_variable = _tk.StringVar(master=self)
                self._caption_trace = source.trace_add('write', self._show_variable)
                self._show_variable()
                options['textvariable'] = self._display_variable
        return super().configure(cnf, **options)

    config = configure

    def _refresh_language(self):
        if self._source_caption is not None:
            super().configure(text=tr(self._source_caption))
        if self._source_variable is not None:
            self._show_variable()


def _text_widget(widget):
    return type(widget.__name__, (_TextWidget, widget), {})


class Combobox(_ttk.Combobox):
    """Bind the displayed choices to an independent, canonical StringVar."""
    def __init__(self, master=None, *, translate_choices=True, **options):
        value = options.pop('textvariable', None)
        self._value = value if value is not None else _tk.StringVar(master=master)
        self._display = _tk.StringVar(master=master)
        self._choices = tuple(options.pop('values', ()))
        self._translate_choices = translate_choices
        self._syncing = False
        super().__init__(master, textvariable=self._display, **options)
        self._source_trace = self._value.trace_add('write', self._show_selection)
        self._display_trace = self._display.trace_add('write', self._select_source)
        self.bind('<Destroy>', self._remove_traces, add='+')
        _widgets.add(self)
        self._refresh_language()

    def _remove_traces(self, _event):
        self._value.trace_remove('write', self._source_trace)
        self._display.trace_remove('write', self._display_trace)

    def _show_selection(self, *_):
        if self._syncing:
            return
        value = self._value.get()
        self._syncing = True
        try:
            self._display.set(self._forward.get(value, value))
        finally:
            self._syncing = False

    def _select_source(self, *_):
        if self._syncing:
            return
        value = self._display.get()
        self._syncing = True
        try:
            self._value.set(self._reverse.get(value, value))
        finally:
            self._syncing = False

    def _refresh_language(self):
        choices = []
        used = set()
        for choice in self._choices:
            label = tr(choice) if self._translate_choices else choice
            shown = label
            number = 2
            while shown in used:
                shown = f'{label} ({number})'
                number += 1
            used.add(shown)
            choices.append(shown)
        self._forward = dict(zip(self._choices, choices))
        self._reverse = dict(zip(choices, self._choices))
        super().configure(values=choices)
        self._show_selection()

    def get(self):
        return self._value.get()

    def set(self, value):
        self._value.set(value)

    def configure(self, cnf=None, **options):
        if isinstance(cnf, dict):
            options = {**cnf, **options}
            cnf = None
        values = options.pop('values', None)
        if values is not None:
            self._choices = tuple(values)
            self._refresh_language()
        if cnf is not None or options or values is None:
            return super().configure(cnf, **options)

    config = configure


class Notebook(_ttk.Notebook):
    def __init__(self, master=None, **options):
        super().__init__(master, **options)
        self._captions = {}
        _widgets.add(self)

    def add(self, child, **options):
        if 'text' in options:
            self._captions[str(child)] = options['text']
            options['text'] = tr(options['text'])
        return super().add(child, **options)

    def _refresh_language(self):
        for tab in self.tabs():
            if tab in self._captions:
                self.tab(tab, text=tr(self._captions[tab]))


class Canvas(_tk.Canvas):
    def __init__(self, master=None, **options):
        super().__init__(master, **options)
        self._captions = {}
        _widgets.add(self)

    def create_text(self, *coordinates, **options):
        text = options.get('text', '')
        options['text'] = tr(text)
        item = super().create_text(*coordinates, **options)
        if isinstance(text, str) and _HAN.search(text):
            self._captions[item] = text
        return item

    def delete(self, *items):
        if self._captions:
            if 'all' in items:
                self._captions.clear()
            else:
                for item in items:
                    for identity in self.find_withtag(item):
                        self._captions.pop(identity, None)
        return super().delete(*items)

    def _refresh_language(self):
        for item, text in self._captions.items():
            self.itemconfigure(item, text=tr(text))


class Toplevel(_tk.Toplevel):
    def title(self, text=None):
        if text is None:
            return super().title()
        self._caption = text
        _widgets.add(self)
        return super().title(tr(text))

    def _refresh_language(self):
        super().title(tr(self._caption))


class Menu(_tk.Menu):
    def __init__(self, master=None, **options):
        super().__init__(master, **options)
        self._labels = {}
        _widgets.add(self)

    def add_command(self, **options):
        caption = options.get('label', '')
        options['label'] = tr(caption)
        super().add_command(**options)
        self._labels[self.index('end')] = caption

    def _refresh_language(self):
        for index, label in self._labels.items():
            self.entryconfigure(index, label=tr(label))


class Treeview(_ttk.Treeview):
    def __init__(self, master=None, **options):
        super().__init__(master, **options)
        self._headings = {}
        _widgets.add(self)

    def heading(self, column, option=None, **options):
        if 'text' in options:
            self._headings[column] = options['text']
            options['text'] = tr(options['text'])
        return super().heading(column, option, **options)

    def insert(self, parent, index, iid=None, **options):
        if 'text' in options:
            options['text'] = tr(options['text'])
        if 'values' in options:
            options['values'] = tuple(tr(value) for value in options['values'])
        return super().insert(parent, index, iid, **options)

    def _refresh_language(self):
        for column, text in self._headings.items():
            super().heading(column, text=tr(text))


class _ModuleView:
    def __init__(self, module, **overrides):
        self._module = module
        vars(self).update(overrides)

    def __getattr__(self, name):
        return getattr(self._module, name)


def _dialog(function, fields):
    def call(*args, **options):
        for field in fields:
            if field in options:
                options[field] = tr(options[field])
        if 'detail' in options:
            options['detail'] = tr(options['detail'])
        if 'filetypes' in options:
            options['filetypes'] = [(tr(name), pattern) for name, pattern in options['filetypes']]
        return function(*(tr(value) for value in args), **options)
    return call


tk = _ModuleView(_tk, Canvas=Canvas, Toplevel=Toplevel, Menu=Menu,
                 **{name: _text_widget(getattr(_tk, name)) for name in ('Label', 'Button')})
ttk = _ModuleView(
    _ttk, Combobox=Combobox, Notebook=Notebook, Treeview=Treeview,
    **{name: _text_widget(getattr(_ttk, name))
       for name in ('Label', 'Button', 'Checkbutton', 'Radiobutton', 'LabelFrame', 'Menubutton')}
)
messagebox = _ModuleView(_messages, **{
    name: _dialog(getattr(_messages, name), ('title', 'message'))
    for name in ('askyesno', 'showerror', 'showinfo', 'showwarning')
})
filedialog = _ModuleView(_files, **{
    name: _dialog(getattr(_files, name), ('title',))
    for name in ('askdirectory', 'askopenfilenames')
})
simpledialog = _ModuleView(_prompts, askstring=_dialog(_prompts.askstring, ('title', 'prompt')))
