"""Check live language changes through the workbench's normal controls."""
import hashlib
import json
import re
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from unittest.mock import patch

from PIL import Image

import ui_language
from workbench_app import App, gpu_compute, widgets


def _wait(root, ready, message):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        root.update()
        if ready():
            return
        time.sleep(.01)
    raise AssertionError(message)


def _choose(app, index):
    app.language_menu.current(index)
    app.change_language()
    app.root.update_idletasks()


def _captions(root):
    for widget in widgets(root):
        for option in ('text', 'textvariable'):
            try:
                value = widget.cget(option)
                if option == 'textvariable' and value:
                    value = root.getvar(str(value))
                yield str(value)
            except tk.TclError:
                pass


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ui_language.PREFERENCE = folder / 'language.json'
    ui_language.set_language('zh')
    photo = folder / '样片.jpg'
    Image.new('RGB', (96, 64), (70, 110, 160)).save(photo)
    original = hashlib.sha256(photo.read_bytes()).hexdigest()
    export = folder / 'export'
    export.mkdir(exist_ok=True)

    with patch.object(gpu_compute, 'warmup', return_value=None):
        root = tk.Tk()
        root.withdraw()
        app = App(root)
        try:
            root.update_idletasks()
            _choose(app, 1)
            han = re.compile('[\u3400-\u9fff]')
            remaining = [text for text in _captions(root) if han.search(text)]
            assert not remaining, remaining
            assert app.tabs.tab(0, 'text').strip() == 'Photos'
            assert app.filter_export_button.cget('text') == 'Export filter files'
            with patch.object(messagebox, 'Message') as dialog:
                app.help()
                options = dialog.call_args.kwargs
                assert options['title'] == 'Help'
                assert not han.search(options['message'])

            app.add_files([str(photo)])
            _wait(root, lambda: app.before is not None, 'Photo preview did not load')
            app.editor_manual_changed('基础调整', exposure=.75)
            app.group_combo.current(1)
            app.build_cards()
            group = app.group.get()
            assert group == '四季'
            app.format.set('PNG')
            app.bits.set('8')
            app.output.set(str(export))
            _choose(app, 0)
            assert app.filter_export_button.cget('text') == '导出滤镜文件'
            _choose(app, 1)
            assert app.group.get() == group
            assert app.group_combo.cget('values')[app.group_combo.current()] == 'Four Seasons'
            assert app.format.get() == 'PNG'
            assert app.editor_state['exposure'] == .75
            assert app.file_list.get(0) == photo.name
            assert app.current_path == str(photo.resolve())
            assert app.callback_error_count == 0

            app.start()
            _wait(root, lambda: not app.busy, 'PNG export did not complete')
            files = list(export.glob('*.png'))
            assert len(files) == 1
            with Image.open(files[0]) as image:
                assert image.size == (96, 64)
            assert hashlib.sha256(photo.read_bytes()).hexdigest() == original
            assert ui_language._read_language() == 'en'
        finally:
            app.close()

        root = tk.Tk()
        root.withdraw()
        app = App(root)
        try:
            assert app.language_choice.get() == 'English'
            assert app.filter_export_button.cget('text') == 'Export filter files'
            assert app.tabs.tab(1, 'text').strip() == 'Video'
            assert app.callback_error_count == 0
        finally:
            app.close()

    (folder / 'language-verification.json').write_text(json.dumps({
        'live_switch': True,
        'english_startup': True,
        'persists_language': True,
        'photo_and_settings_preserved': True,
        'help_translated': True,
        'export_verified': True,
    }, indent=2), encoding='utf-8')
