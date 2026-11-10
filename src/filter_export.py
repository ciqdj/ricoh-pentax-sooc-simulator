"""Export verified, explicitly sampled standard looks, never the legacy bundle."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import shutil
import tempfile
import uuid
import numpy as np
from app_paths import ROOT
from filter_catalog import require_renderer, PENTAX, RICOH, CREATIVE_LOOKS, PENTAX_LOOKS, GR_LOOKS,GR_ENGLISH
from native_backend import LOOKS
from lut_formats import write_cube, write_xmp
from lut_formats import grid
from engine import apply as creative_apply

DATA = ROOT / 'data' / 'standard-filter-luts.npz'
MANIFEST = ROOT / 'data' / 'standard-filter-luts.json'
DESCRIPTION = ('由 DCU 5.10.2.2 原厂代码、K-1 标准参数采样生成的 33³ sRGB LUT。'
               '它是固定色彩映射，不能完整保留照片引擎的局部处理、锐化及随照片变化的处理；'
               '不包含个人预设、临时曝光和浓淡调整。')


def load_tables():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('滤镜资源校验失败，请重新取得完整程序。')
    with np.load(DATA, allow_pickle=False) as archive:
        if set(archive.files) != set(LOOKS): raise ValueError('标准滤镜资源不完整。')
        tables = {key: archive[key] for key in LOOKS}
    for table in tables.values():
        if table.shape != (33, 33, 33, 3) or not np.isfinite(table).all() or table.min() < 0 or table.max() > 1:
            raise ValueError('滤镜数据无效。')
    sample=grid(33)
    tables.update({key:creative_apply(sample,key) for key in CREATIVE_LOOKS})
    return tables


def export_filters(folder, brand=PENTAX, formats=('cube', 'xmp'), progress=None):
    formats = tuple(dict.fromkeys(formats))
    if not formats or set(formats) - {'cube', 'xmp'}: raise ValueError('请选择导出格式。')
    if brand not in (PENTAX,RICOH):
        return export_custom_filters(folder,brand,formats,progress)
    require_renderer(brand)
    if brand==RICOH:return export_gr4_filters(folder,formats,progress)
    tables = load_tables()
    parent = Path(folder).resolve()
    if not parent.is_dir(): raise ValueError('请选择已存在的保存文件夹。')
    name = 'PENTAX_滤镜文件_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
    dest = parent / name
    files = []
    # Publish only after every requested format succeeds. TemporaryDirectory
    # owns only this freshly-created child of the user's chosen directory.
    with tempfile.TemporaryDirectory(prefix='.filter-export-', dir=parent) as staging:
        stage = Path(staging).resolve()
        if stage.parent != parent: raise ValueError('Invalid export staging directory')
        payload = stage / 'payload'
        payload.mkdir()
        for index, (key, table) in enumerate(tables.items()):
            creative=key in CREATIVE_LOOKS
            if progress: progress(index / len(tables) * 100, f'导出 {index+1}/{len(tables)} · {key}')
            if 'cube' in formats:
                target = payload / 'Photoshop_CUBE' / f'PENTAX_{key}.cube'
                write_cube(target, table, 'PENTAX ' + key + ' sampled sRGB',
                           'Restored 3.1 creative formula; sampled sRGB.' if creative else '33^3 sampled DCU K-1 standard look; sRGB in/out. Not the full photo engine.')
                files.append(target.relative_to(payload).as_posix())
            if 'xmp' in formats:
                target = payload / 'Lightroom_Classic_XMP' / f'PENTAX_{key}.xmp'
                write_xmp(target, table, f'PENTAX {key} · {PENTAX_LOOKS[key]["name"]}',
                          '3.1 创意配方的固定 LUT；非完整原厂处理。' if creative else DESCRIPTION,
                          group='賓得 · 3.1 创意滤镜' if creative else '賓得 · 标准滤镜 LUT')
                files.append(target.relative_to(payload).as_posix())
        readme = ('导出内容：宾得 17 个标准滤镜（包含四季），以及恢复的 3.1 黑白变体、创意组合和基调曲线，共 34 个。\n\n' + DESCRIPTION + '\n\n'
                  'Photoshop：图像先转换到 sRGB；添加“颜色查找”调整图层，在 3D LUT 中载入 .cube。\n'
                  'Lightroom Classic：导入 Lightroom_Classic_XMP 中的 .xmp，随后在配置文件浏览器的“賓得 · 标准滤镜 LUT”分组选择。\n'
                  '这些 XMP 是配置文件，不是开发预设滑块，也不会把其他品牌 RAW 变成宾得传感器原始数据。\n'
                  '宿主的软件版本、输入显影与工作色彩空间会影响结果；不能据此保证与原厂显影逐像素一致。\n')
        (payload / '使用说明.txt').write_text(readme, encoding='utf-8')
        hashes = {name: hashlib.sha256((payload / name).read_bytes()).hexdigest() for name in files}
        (payload / 'manifest.json').write_text(json.dumps({
            'brand': brand, 'look_count': len(tables), 'formats': list(formats),
            'sampled_approximation': True, 'complete_native_engine': False,
            'standard_settings_only': True, 'input_colour_space': 'sRGB',
            'files': hashes}, indent=2), encoding='utf-8')
        payload.rename(dest)
    if progress: progress(100, '滤镜文件导出完成')
    return dest, files


def gr4_tables():
    from gr4_renderer import KEYS,tone_count,point
    sample=grid(33);tables={};names={}
    for key in KEYS:
        count=tone_count(key)
        for tone in range(count):
            name=key+(f'_Tone{tone}' if count>1 else '')
            tables[name]=point(sample,key,tone)
            names[name]=GR_ENGLISH.get(key,key)+' · '+GR_LOOKS[key]['name']+(f' · 色调 {tone}' if count>1 else '')
    return tables,names


def export_gr4_filters(folder,formats,progress=None):
    parent=Path(folder).resolve()
    if not parent.is_dir():raise ValueError('请选择已存在的保存文件夹。')
    tables,names=gr4_tables()
    dest=parent/('GR4_滤镜文件_'+datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:6]);files=[]
    description='GR4 色彩模型 R3 的 33³ sRGB 采样；使用普通 sRGB 输入解码、原厂曲线参数及模式矩阵，覆盖色彩处理阶段。'
    with tempfile.TemporaryDirectory(prefix='.filter-export-',dir=parent) as staging:
        stage=Path(staging).resolve()
        if stage.parent!=parent:raise ValueError('Invalid export staging directory')
        payload=stage/'payload';payload.mkdir()
        for index,(name,table) in enumerate(tables.items()):
            if progress:progress(index/len(tables)*100,f'导出理光 {index+1}/{len(tables)} · {names[name]}')
            if 'cube' in formats:
                target=payload/'Photoshop_CUBE'/f'GR4_{name}.cube'
                write_cube(target,table,'GR4 '+name+' R3 model sRGB','33^3 sampled partial research model, not the complete original camera ISP.')
                files.append(target.relative_to(payload).as_posix())
            if 'xmp' in formats:
                target=payload/'Lightroom_Classic_XMP'/f'GR4_{name}.xmp'
                write_xmp(target,table,'GR4 '+names[name],description,group='理光 GR4 · 色彩研究模型')
                files.append(target.relative_to(payload).as_posix())
        readme=('理光 GR4 · 色彩研究模型滤镜文件\n\n'+description+'\n\n'
                '11 种模式、17 组色调设置；电影黄调、电影绿调、交叉处理各有 0/1/2 三档。\n'
                '不包含漂白旁路、复古、HDR；不包含临时曝光、浓淡或个人预设。\n'
                'Photoshop：先将图像转换到 sRGB，在“颜色查找”调整图层载入 CUBE。\n'
                'Lightroom Classic：导入 XMP 后，在“理光 GR4 · 色彩研究模型”配置文件分组中选择；XMP 不是开发预设滑块。\n'
                '宿主显影和色彩空间会影响结果；未进行 Adobe 宿主导入验收。不能将其他相机 RAW 变成理光传感器原始数据。\n')
        (payload/'使用说明.txt').write_text(readme,encoding='utf-8')
        (payload/'manifest.json').write_text(json.dumps({'brand':RICOH,'look_count':11,'setting_count':17,'formats':list(formats),
            'sampled_approximation':True,'complete_native_engine':False,'standard_settings_only':True,'input_colour_space':'sRGB',
            'files':{name:hashlib.sha256((payload/name).read_bytes()).hexdigest() for name in files}},indent=2),encoding='utf-8')
        payload.rename(dest)
    if progress:progress(100,'理光滤镜文件导出完成')
    return dest,files


def export_custom_filters(folder,brand,formats=('cube',),progress=None):
    """Copy the imported CUBEs without resampling or attaching user paths."""
    from custom_lut import get_library
    if tuple(formats)!=('cube',):raise ValueError('自定义滤镜目前支持导出原始 CUBE 文件。')
    library=get_library()
    brands=library.brands()
    if brand not in brands:raise ValueError('未知自定义品牌')
    looks=library.catalog(brand)
    if not looks:raise ValueError('这个品牌还没有导入滤镜。')
    parent=Path(folder).resolve()
    if not parent.is_dir():raise ValueError('请选择已存在的保存文件夹。')
    dest=parent/('CUBE_滤镜文件_'+datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:6])
    files=[];hashes={}
    with tempfile.TemporaryDirectory(prefix='.filter-export-',dir=parent) as staging:
        stage=Path(staging).resolve()
        if stage.parent!=parent:raise ValueError('Invalid export staging directory')
        payload=stage/'payload';payload.mkdir()
        for index,(key,record) in enumerate(looks.items()):
            title=str(record.get('name',record.get('title','LUT')))
            safe=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',title).strip(' .')[:90] or 'LUT'
            name=f'{index+1:03d}_{safe}.cube'
            if progress:progress(100*index/len(looks),f'导出 {index+1}/{len(looks)} · {title}')
            source=library.get_cube_path(key);target=payload/name
            with source.open('rb') as handle:before=hashlib.file_digest(handle,'sha256').hexdigest()
            shutil.copyfile(source,target)
            with target.open('rb') as handle:after=hashlib.file_digest(handle,'sha256').hexdigest()
            if before!=after:raise RuntimeError('CUBE 导出期间文件发生变化，请重试。')
            files.append(name);hashes[name]=after
        (payload/'使用说明.txt').write_text(
            '自定义 CUBE 滤镜\n\n保留导入文件的原始内容，没有重新采样。\n'
            '不包含临时曝光、浓淡、降噪或手动调整。宿主的输入编码应符合各 LUT 的要求。\n',encoding='utf-8')
        (payload/'manifest.json').write_text(json.dumps({'format':'cube','original_bytes_preserved':True,
            'look_count':len(files),'files':hashes},ensure_ascii=False,indent=2),encoding='utf-8')
        if hasattr(library,'export_provenance'):library.export_provenance(brand,payload)
        payload.rename(dest)
    if progress:progress(100,'CUBE 滤镜导出完成')
    return dest,files
