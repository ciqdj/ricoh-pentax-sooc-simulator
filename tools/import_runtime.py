"""Extract immutable runtime resources from this project's Windows release."""
from pathlib import Path,PurePosixPath
import argparse
from PyInstaller.archive.readers import CArchiveReader

def main():
    p=argparse.ArgumentParser();p.add_argument('release',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]/'src'
    archive=CArchiveReader(str(a.release));count=0
    prefixes=('native/','neural-runtime/','denoise-data/','rawpy_f/','bin/')
    exact={'assets.zip','resource-index.json'}
    for archived_name in archive.toc:
        name=archived_name.replace('\\','/');part=PurePosixPath(name)
        if part.is_absolute() or '..' in part.parts or ':' in name:continue
        if name not in exact and not name.startswith(prefixes):continue
        target=root.joinpath(*part.parts)
        if target.exists():continue
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.extract(archived_name));count+=1
    print(f'Imported {count} runtime resources into {root}')

if __name__=='__main__':main()
