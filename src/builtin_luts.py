"""Read-only bundled CUBEs alongside the user's independently writable library."""
from pathlib import Path
import json,shutil
from custom_lut import Library


class BundledLibrary(Library):
    def create_brand(self,*args,**kwargs):
        raise ValueError('内置滤镜库不可修改，请新建自定义品牌。')
    create_category=create_brand
    import_cube=create_brand
    import_files=create_brand


class CombinedLibrary:
    def __init__(self,user,root):
        self.user=user;self.root=user.root
        self.bundled=BundledLibrary(root)
        self._brands=self.bundled.brands()
        self._keys=frozenset(self.bundled.catalog())
        source=Path(root)/'sources.json'
        self.sources=json.loads(source.read_text(encoding='utf-8'))['brands'] if source.exists() else {}

    def __getattr__(self,name):return getattr(self.user,name)
    def is_builtin_brand(self,brand):return brand in self._brands
    def brands(self):return {**self._brands,**self.user.brands()}
    def categories(self,brand):
        return (self.bundled if self.is_builtin_brand(brand) else self.user).categories(brand)
    def catalog(self,brand_id=None,category_id=None):
        if brand_id is not None:
            return (self.bundled if self.is_builtin_brand(brand_id) else self.user).catalog(brand_id,category_id)
        return {**self.bundled.catalog(category_id=category_id),**self.user.catalog(category_id=category_id)}
    def lookup(self,key):return (self.bundled if key in self._keys else self.user).lookup(key)
    def get_cube(self,key,cancelled=None):return (self.bundled if key in self._keys else self.user).get_cube(key,cancelled)
    def get_cube_path(self,key):return (self.bundled if key in self._keys else self.user).get_cube_path(key)
    def records(self,brand_id=None):return list(self.catalog(brand_id).values())
    def list_brands(self):return [{'id':key,'name':value} for key,value in self.brands().items()]
    def list_categories(self,brand_id):return [{'id':key,'brand_id':brand_id,'name':value} for key,value in self.categories(brand_id).items()]
    def list_luts(self,brand_id=None,category_id=None):return list(self.catalog(brand_id,category_id).values())
    def create_brand(self,name):return self.user.create_brand(name)
    add_brand=create_brand
    def _writable(self,brand):
        if self.is_builtin_brand(brand):raise ValueError('这是内置滤镜库。请在“管理”中新建自己的品牌，再导入 CUBE。')
    def create_category(self,brand_id,name):
        self._writable(brand_id);return self.user.create_category(brand_id,name)
    add_group=create_category
    def import_cube(self,path,brand_id,category_id=None,name=None,cancelled=None):
        self._writable(brand_id);return self.user.import_cube(path,brand_id,category_id,name,cancelled)
    def import_files(self,paths,brand_id,category_id=None,cancelled=None,progress=None):
        self._writable(brand_id);return self.user.import_files(paths,brand_id,category_id,cancelled,progress)
    def source_note(self,brand):
        info=self.sources.get(brand)
        return (info['label']+'\n'+info['input_note']+'\n'+info['repository']+'\n'+info['license']) if info else ''
    def export_provenance(self,brand,destination):
        info=self.sources.get(brand)
        if not info:return
        target=Path(destination)/'来源与许可';target.mkdir()
        for item in (self.bundled.root/'sources').iterdir():
            if item.name=='NOTICE.txt' or item.name.startswith(brand):shutil.copyfile(item,target/item.name)
        (target/'来源.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')


def with_bundled(user):
    from app_paths import ROOT
    root=ROOT/'builtin-luts'
    return CombinedLibrary(user,root) if (root/'index.json').is_file() else user
