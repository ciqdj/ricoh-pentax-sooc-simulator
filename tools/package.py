"""Build a single-file Windows payload from a prepared source checkout."""
from pathlib import Path
import argparse,subprocess,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];src=root/'src';work=a.work.resolve();out=a.output.resolve()
    work.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    args=[sys.executable,'-m','PyInstaller','--noconfirm','--onefile','--windowed',
        '--name','RICOH-PENTAX-SOOC-Simulator','--distpath',str(out),'--workpath',str(work/'pyinstaller'),
        '--specpath',str(work),'--icon',str(src/'branding/app.ico')]
    for name in ('data','native-data','gr4-data','builtin-luts','licenses','denoise-data','neural-runtime','branding'):
        path=src/name
        if path.is_dir():args+=['--add-data',str(path)+';'+name]
    for name in ('assets.zip','resource-index.json'):
        if (src/name).exists():args+=['--add-data',str(src/name)+';.']
    args+=['--add-data',str(src/'gr4-core/photo_kernels.cl')+';gr4-core']
    for folder in ('native','gr4-core','rawpy_f','bin'):
        for path in (src/folder).glob('*'):
            if path.suffix.lower() in ('.dll','.exe','.pyd'):args+=['--add-binary',str(path)+';'+folder]
    for name in ('native_worker','verify_gr4_controls','verify_ui_language','imagecodecs._png','imagecodecs._shared_cython',
                 'imagecodecs._deflate','imagecodecs._zlib','imagecodecs._imcd','rawpy_f._rawpy'):
        args+=['--hidden-import',name]
    args+=[str(src/'workbench_entry.py')]
    subprocess.run(args,cwd=root,check=True)

if __name__=='__main__':main()
