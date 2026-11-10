"""Render a normalized SDR Rec.709 / gamma 2.4 video through an exported look.

This deliberately requires explicit input-space confirmation. Log, HLG, and PQ
must first be normalized in a colour-managed editor; tagging them 709 is not a conversion.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,tempfile,sys
from app_paths import resource_bytes, resource_file, writable_dir, ffmpeg_path

def render(source,dest,look,input_space='rec709-g24',input_range='auto',codec='prores',log=None):
    if input_space!='rec709-g24':raise ValueError('请先将 Log/HDR 转换为 Rec.709 Gamma 2.4。')
    if codec not in ('prores','hevc10'):raise ValueError('不支持的视频编码。')
    source=Path(source).resolve();dest=Path(dest).resolve()
    if source==dest or dest.exists():raise ValueError('输出必须是新文件，不能覆盖原视频。')
    if not source.is_file():raise ValueError('找不到输入视频。')
    manifest=json.loads(resource_bytes('exports/manifest.json'))
    item=next((x for x in manifest['looks'] if x['id']==look),None)
    if not item:raise ValueError('未知风格。')
    cube=resource_file(next(x for x in item['files'] if 'Video_Rec709' in x))
    ffmpeg=ffmpeg_path()
    if codec=='prores' and dest.suffix.lower()!='.mov':raise ValueError('ProRes 输出需使用 .mov 后缀。')
    if codec=='hevc10' and dest.suffix.lower() not in ['.mov','.mp4']:raise ValueError('HEVC 输出需使用 .mp4 或 .mov。')
    dest.parent.mkdir(parents=True,exist_ok=True)
    inrange={'limited':'tv','full':'full','auto':'auto'}[input_range]
    with tempfile.TemporaryDirectory(prefix='.Pentax-video-',dir=dest.parent) as tmp:
        work=Path(tmp).resolve();assert work.parent==dest.parent
        staged=work/('output'+dest.suffix)
        shutil.copyfile(cube,work/'lut.cube')
        pix='yuv422p10le' if codec=='prores' else 'yuv420p10le'
        vf=f'scale=in_range={inrange}:out_range=full:in_color_matrix=bt709,format=gbrpf32le,lut3d=file=lut.cube:interp=tetrahedral,scale=in_range=full:out_range=tv:out_color_matrix=bt709,format={pix},setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709'
        cmd=[ffmpeg,'-hide_banner','-nostdin','-n','-i',str(source),'-map','0:v:0','-map','0:a?',
             '-map_metadata','-1','-map_metadata:s','-1','-map_chapters','-1','-vf',vf]
        if codec=='prores':cmd+=['-c:v','prores_ks','-profile:v','3','-c:a','pcm_s24le']
        else:cmd+=['-c:v','libx265','-preset','medium','-crf','18','-tag:v','hvc1','-c:a','aac','-b:a','256k']
        cmd+=['-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-movflags','+write_colr',str(staged)]
        kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if sys.platform=='win32' else {}
        p=subprocess.Popen(cmd,cwd=work,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',**kwargs)
        tail=[]
        for line in p.stderr:
            tail.append(line.rstrip());tail=tail[-30:]
            if log:log(line.rstrip())
        if p.wait():raise RuntimeError('\n'.join(tail))
        # On Windows rename refuses an existing destination. Publish only the
        # fully encoded file, on the same volume, with no partial user output.
        if dest.exists():raise ValueError('输出文件已存在，未覆盖。')
        staged.rename(dest)
    return {'look':look,'input_space':input_space,'codec':codec,'personal_metadata_copied':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--look',required=True);p.add_argument('--input-space',required=True,choices=['rec709-g24'])
    p.add_argument('--range',default='auto',choices=['auto','limited','full'])
    p.add_argument('--codec',default='prores',choices=['prores','hevc10'])
    args=p.parse_args()
    report=render(args.input,args.output,args.look,args.input_space,args.range,args.codec,log=print)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
