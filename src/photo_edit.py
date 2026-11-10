"""Photo-owned, parameter-only edit history and explicit preset commits.

Selecting an old step never deletes anything or changes the committed card
basis. A new operation from that step requires an explicit branch decision.
No image arrays, decoded pixels, or model sessions belong in this module.
"""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import hashlib,json,math,os,re,threading,time,uuid
from pathlib import Path

from postprocess import validate as validate_post
from color_balance import validate as validate_balance


class BranchRequired(ValueError):
    pass


_IDENTIFIER=re.compile(r'^[0-9a-f]{32}$')


def _text(value,name,limit=512):
    if not isinstance(value,str) or not value or len(value)>limit or any(ord(c)<32 for c in value):
        raise ValueError(name+'格式无效')
    return value


def _time(value):
    _text(value,'历史时间',80)
    try:parsed=datetime.fromisoformat(value)
    except ValueError as exc:raise ValueError('历史时间格式无效') from exc
    if parsed.tzinfo is None:raise ValueError('历史时间缺少时区')
    return value


def default_state():
    return {'exposure':0.,'balance':[0.,0.],'post':validate_post(),
            'filter':{'brand':'pentax','look':None,'amount':1.,'controls':{}},
            'denoise':{'kind':'off','model':'nafnet','amount':100.,'native':{}}}


def _number(value,low,high):
    if type(value) not in (int,float) or not low<=value<=high or not math.isfinite(value):
        raise ValueError('编辑参数超出有效范围')
    return float(value)


def _controls(value):
    try:
        valid=isinstance(value,dict) and all(isinstance(k,str) and type(v) in (bool,int,float) and
                                           math.isfinite(v) for k,v in value.items())
    except (TypeError,OverflowError):valid=False
    if not valid:raise ValueError('滤镜参数格式无效')
    return dict(value)


def normalize(value):
    if not isinstance(value,dict):raise ValueError('编辑记录格式无效')
    result=default_state()
    if set(value)-set(result):raise ValueError('编辑记录包含未知字段')
    result['exposure']=_number(value.get('exposure',0),-5,5)
    try:result['balance']=list(validate_balance(value.get('balance',(0,0))))
    except (TypeError,OverflowError) as exc:raise ValueError('色彩平衡参数格式无效') from exc
    if value.get('post') is not None and not isinstance(value['post'],dict):raise ValueError('调色参数格式无效')
    try:result['post']=validate_post(value.get('post'))
    except (TypeError,KeyError,OverflowError) as exc:raise ValueError('调色参数格式无效') from exc
    # AI belongs to the input stage, never to this manual colour operation.
    result['post'].update(denoise=0.,denoise_model='dcu',noise_level=15.)
    if not isinstance(value.get('filter',{}),dict):raise ValueError('滤镜选择格式无效')
    f=dict(result['filter'],**value.get('filter',{}))
    if set(f)!=set(result['filter']) or f['brand'] not in ('pentax','ricoh_gr4','custom'):
        raise ValueError('滤镜选择格式无效')
    if f['look']=='':f['look']=None
    if f['look'] is not None:_text(f['look'],'滤镜名称',256)
    f['amount']=_number(f['amount'],0,2);f['controls']=_controls(f['controls'])
    from native_controls import NR_CONTROLS
    f['controls']={k:v for k,v in f['controls'].items() if k not in NR_CONTROLS}
    result['filter']=f
    if not isinstance(value.get('denoise',{}),dict):raise ValueError('降噪选择格式无效')
    d=dict(result['denoise'],**value.get('denoise',{}))
    if set(d)!=set(result['denoise']) or d['kind'] not in ('off','native','ai'):
        raise ValueError('降噪选择格式无效')
    # Retired SCUNet must not silently start a different long-running model.
    if d['model']=='scunet':
        d['model']='nafnet'
        if d['kind']=='ai':d['kind']='off'
    if d['model'] not in ('nafnet','restormer'):raise ValueError('AI 模型无效')
    d['amount']=_number(d['amount'],0,100);d['native']=_controls(d['native'])
    if set(d['native'])-set(NR_CONTROLS):raise ValueError('原厂降噪参数无效')
    for key,number in d['native'].items():
        spec=NR_CONTROLS[key]
        if type(number) is not int or not spec['min']<=number<=spec['max']:raise ValueError('原厂降噪参数超出范围')
    result['denoise']=d
    return result


# The GUI uses this descriptive name; retain normalize for existing callers.
validate_state=normalize


def manual_state(value):
    state=normalize(value)
    return {name:deepcopy(state[name]) for name in ('exposure','balance','post')}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,
                                    separators=(',',':')).encode('utf-8')).hexdigest()


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='milliseconds')


def _changed_paths(before,after,prefix=()):
    paths=[]
    for key in before.keys()|after.keys():
        a,b=before.get(key),after.get(key)
        if a==b:continue
        if isinstance(a,dict) and isinstance(b,dict):paths.extend(_changed_paths(a,b,prefix+(key,)))
        else:paths.append(prefix+(key,))
    return frozenset(paths)


def _write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        # The C encoder emits the complete small parameter document once.
        # json.dump's many Python stream writes made a 1,000-step slider edit
        # block the Tk thread for over 100 ms on the development machine.
        encoded=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        if len(encoded.encode('utf-8'))>64*1024*1024:raise ValueError('照片历史文件过大，已保留上次保存的完整历史')
        with temporary.open('x',encoding='utf-8') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


class PhotoHistory:
    def __init__(self,path=None,identity=None,data=None):
        self.path=Path(path) if path is not None else None
        self.identity=deepcopy(identity)
        self._lock=threading.RLock()
        self._last_record=None
        if data is None:
            step={'id':uuid.uuid4().hex,'time':timestamp(),'title':'原始照片','state':default_state()}
            self.steps=[step];self.cursor=0;self.commits=[];self.commit_serial=0
            self._committed_base=manual_state(step['state']);self._committed_styles={}
        else:self._load(data)

    def _load(self,data):
        # Loading is read-only with respect to the caller and existing file.
        data=deepcopy(data)
        required={'version','identity','steps','cursor','commits','commit_serial','committed_base'}
        if not isinstance(data,dict) or not required<=set(data) or set(data)-required-{'committed_styles'}:
            raise ValueError('历史记录结构无效，已保留原数据')
        if type(data['version']) is not int or data['version']!=1 or data['identity']!=self.identity:raise ValueError('历史与照片不匹配')
        steps=data['steps']
        if not isinstance(steps,list) or not steps:raise ValueError('历史记录为空')
        ids=set()
        for row in steps:
            if not isinstance(row,dict) or set(row)!={'id','time','title','state'} or not isinstance(row.get('id'),str) or not _IDENTIFIER.fullmatch(row['id']) or row['id'] in ids:
                raise ValueError('历史步骤标识无效')
            ids.add(row['id']);row['state']=normalize(row['state'])
            _text(row['title'],'历史步骤标题');_time(row['time'])
        cursor=data['cursor']
        if type(cursor) is not int or not 0<=cursor<len(steps):raise ValueError('历史位置无效')
        commits=data['commits']
        if not isinstance(commits,list):raise ValueError('提交记录格式无效')
        if type(data['commit_serial']) is not int or data['commit_serial']<0:raise ValueError('提交编号无效')
        prior=-1;prior_serial=0;group_ids=set()
        positions={row['id']:i for i,row in enumerate(steps)}
        for group in commits:
            if not isinstance(group,dict) or set(group)!={'id','time','title','end_step','serial'}:raise ValueError('提交记录格式无效')
            if not isinstance(group['id'],str) or not _IDENTIFIER.fullmatch(group['id']) or group['id'] in group_ids:raise ValueError('提交步骤标识无效')
            if not isinstance(group['end_step'],str):raise ValueError('提交边界无效')
            end=positions.get(group['end_step'],-1)
            if end<=prior:raise ValueError('提交边界无效')
            if type(group['serial']) is not int or not prior_serial<group['serial']<=data['commit_serial']:raise ValueError('提交编号无效')
            _text(group['title'],'提交标题');_time(group['time'])
            prior=end;prior_serial=group['serial'];group_ids.add(group['id'])
        self.steps=deepcopy(steps);self.cursor=cursor;self.commits=deepcopy(commits)
        self.commit_serial=data['commit_serial']
        if not isinstance(data['committed_base'],dict) or set(data['committed_base'])-{'exposure','balance','post'}:
            raise ValueError('已提交的调色基础格式无效')
        self._committed_base=manual_state(data['committed_base'])
        self._committed_styles=deepcopy(data.get('committed_styles',{}))
        if not isinstance(self._committed_styles,dict):raise ValueError('已提交滤镜参数格式无效')
        for key,value in self._committed_styles.items():
            style=normalize({'filter':value})['filter']
            if style['look'] is None or key!=self.style_key(style['brand'],style['look']):raise ValueError('已提交滤镜参数标识无效')
            self._committed_styles[key]=style

    @property
    def current(self):
        with self._lock:return deepcopy(self.steps[self.cursor]['state'])

    @property
    def current_id(self):
        with self._lock:return self.steps[self.cursor]['id']

    @property
    def committed_base(self):
        with self._lock:return deepcopy(self._committed_base)

    @property
    def committed_styles(self):
        with self._lock:return deepcopy(self._committed_styles)

    @property
    def is_previewing(self):return self.cursor<len(self.steps)-1

    @property
    def can_undo(self):return self.cursor>0

    @property
    def can_redo(self):return self.cursor<len(self.steps)-1

    @staticmethod
    def style_key(brand,look):return brand+'\n'+str(look)

    def snapshot(self):
        with self._lock:
            return {'version':1,'identity':deepcopy(self.identity),'steps':deepcopy(self.steps),
                    'cursor':self.cursor,'commits':deepcopy(self.commits),'commit_serial':self.commit_serial,
                    'committed_base':self.committed_base,'committed_styles':self.committed_styles}

    def save(self):
        # No mutable reference escapes this lock. Public snapshots still deep
        # copy, but persisting needs neither a second nor third image of every
        # old parameter dictionary on each mouse movement.
        with self._lock:
            if self.path is not None:
                _write_json(self.path,{'version':1,'identity':self.identity,'steps':self.steps,
                    'cursor':self.cursor,'commits':self.commits,'commit_serial':self.commit_serial,
                    'committed_base':self._committed_base,'committed_styles':self._committed_styles})

    @contextmanager
    def _transaction(self):
        # Atomic replace protects the file. Rollback also protects the live
        # cursor, future branch and committed card basis when saving fails.
        with self._lock:
            # Records/groups are immutable once installed: edits replace or
            # append a row, never mutate an existing row/state in place.
            old=(list(self.steps),self.cursor,list(self.commits),self.commit_serial,
                 self._committed_base,dict(self._committed_styles),self._last_record)
            try:
                yield
                self.save()
            except BaseException:
                (self.steps,self.cursor,self.commits,self.commit_serial,
                 self._committed_base,self._committed_styles,self._last_record)=old
                raise

    def _branch(self,allow):
        if type(allow) is not bool:raise ValueError('历史分支确认必须是明确的是或否')
        if not self.is_previewing:return
        if not allow:raise BranchRequired('从此步骤继续编辑将删除后续历史，是否继续？')
        kept={row['id'] for row in self.steps[:self.cursor+1]}
        self.steps=self.steps[:self.cursor+1]
        self.commits=[row for row in self.commits if row['end_step'] in kept]
        # The card basis is deliberately NOT rolled back here. Only Commit
        # changes it, even if the associated old history branch is discarded.
        self._last_record=None

    def record(self,state,title='调整',allow_branch=False,coalesce_key=None):
        return self._record(state,title,allow_branch,coalesce_key,False)

    def _record(self,state,title,allow_branch,coalesce_key,force):
        state=normalize(state)
        title=_text(title,'历史步骤标题')
        if coalesce_key is not None:_text(coalesce_key,'连续调整标识',120)
        if type(allow_branch) is not bool:raise ValueError('历史分支确认必须是明确的是或否')
        with self._lock:
            if not force and state==self.steps[self.cursor]['state']:return False
            changed=_changed_paths(self.steps[self.cursor]['state'],state)
            with self._transaction():
                self._branch(allow_branch)
                now=time.monotonic_ns();previous=self._last_record
                last_committed={row['end_step'] for row in self.commits}
                coalesce=(not force and coalesce_key is not None and previous is not None and previous[0]==coalesce_key
                          and 0<=now-previous[1]<450_000_000 and previous[2]==self.current_id and previous[3]==changed
                          and self.current_id not in last_committed and self.cursor>0)
                row={'id':self.current_id if coalesce else uuid.uuid4().hex,'time':timestamp(),'title':title,'state':state}
                if coalesce:self.steps[self.cursor]=row
                else:self.steps.append(row);self.cursor=len(self.steps)-1
                self._last_record=(coalesce_key,now,self.current_id,changed)
            return True

    def preview(self,step_id):
        with self._transaction():
            index=next((i for i,row in enumerate(self.steps) if row['id']==step_id),None)
            if index is None:raise ValueError('该历史步骤已不存在')
            self.cursor=index;self._last_record=None
        return self.current

    def undo(self):
        with self._transaction():
            if self.can_undo:self.cursor-=1
            self._last_record=None
        return self.current

    def redo(self):
        with self._transaction():
            if self.can_redo:self.cursor+=1
            self._last_record=None
        return self.current

    def commit(self,allow_branch=False,title=None):
        if title is not None:_text(title,'提交标题')
        with self._transaction():
            was_previewing=self.is_previewing
            self._branch(allow_branch)
            if self.commits and self.commits[-1]['end_step']==self.current_id:
                if not was_previewing:
                    # Re-clicking Commit cannot manufacture an empty group.
                    self._last_record=None
                    return deepcopy(self.commits[-1])
                # Explicitly committing an old committed position after a
                # confirmed branch gets a parameter snapshot of its own.
                self.steps.append({'id':uuid.uuid4().hex,'time':timestamp(),
                                   'title':'提交时的参数','state':self.current})
                self.cursor=len(self.steps)-1
            state=self.current;self.commit_serial+=1
            self._committed_base=manual_state(state)
            style=state['filter']
            if style['look'] is not None:
                self._committed_styles[self.style_key(style['brand'],style['look'])]=deepcopy(style)
            row={'id':uuid.uuid4().hex,'time':timestamp(),'title':title or f'提交 {self.commit_serial}',
                 'end_step':self.current_id,'serial':self.commit_serial}
            self.commits.append(row);self._last_record=None
        return deepcopy(row)

    def restore_original(self,allow_branch=False):
        return self._record(default_state(),'恢复原始',allow_branch,None,True)

    def groups(self,include_states=True):
        """Ordered UI groups, each containing its actual parameter steps."""
        with self._lock:return self._groups(include_states)

    def _groups(self,include_states=True):
        result=[];start=0
        def rows(values):
            if include_states:return deepcopy(values)
            return [{key:row[key] for key in ('id','time','title')} for row in values]
        positions={step['id']:i for i,step in enumerate(self.steps)}
        for commit in self.commits:
            end=positions[commit['end_step']]
            result.append(dict(deepcopy(commit),steps=rows(self.steps[start:end+1]),committed=True))
            start=end+1
        if start<len(self.steps):
            result.append({'id':'pending','time':self.steps[start]['time'],'title':'未提交',
                           'end_step':self.steps[-1]['id'],'steps':rows(self.steps[start:]),'committed':False})
        return result


class PhotoEditStore:
    def __init__(self,root=None):
        if root is None:
            from app_paths import CONFIG
            root=CONFIG/'photo_edits'
        self.root=Path(root);self._histories={};self._lock=threading.RLock()

    def for_photo(self,path):
        from photo_denoise import identity_for
        identity=asdict(identity_for(path));key=fingerprint(identity)
        with self._lock:
            if key not in self._histories:
                file=self.root/(key+'.json');data=None
                if file.exists():
                    if file.stat().st_size>64*1024*1024:raise ValueError('照片历史文件过大，已保留原数据')
                    def object_pairs(pairs):
                        result={}
                        for key,value in pairs:
                            if key in result:raise ValueError('duplicate key')
                            result[key]=value
                        return result
                    def invalid_number(value):raise ValueError('non-finite number')
                    try:
                        data=json.loads(file.read_text(encoding='utf-8'),object_pairs_hook=object_pairs,parse_constant=invalid_number)
                        if not isinstance(data,dict):raise ValueError('history must be an object')
                    except (UnicodeError,ValueError) as exc:raise ValueError('照片历史文件损坏，已保留原数据，没有重置') from exc
                self._histories[key]=PhotoHistory(file,identity,data)
            return self._histories[key]


_LABELS={'exposure':'曝光','balance':'色彩平衡','contrast':'对比度','saturation':'饱和度',
         'vibrance':'自然饱和度','warmth':'冷暖','tint':'色调','clarity':'清晰度','texture':'纹理',
         'dehaze':'去朦胧','sharpen':'锐化','soften':'柔化','highlights':'高光','shadows':'阴影',
         'blacks':'黑色','whites':'白色','points':'曲线','amount':'浓淡','denoise':'降噪'}

def describe_change(before,after):
    before,after=normalize(before),normalize(after)
    def value(number):return f'{number:+g}' if type(number) in (int,float) else str(number)
    if before['filter']!=after['filter']:
        f=after['filter']
        if f['look']!=before['filter']['look'] or f['brand']!=before['filter']['brand']:
            return '原图' if f['look'] is None else '滤镜 '+f['look']
        if f['amount']!=before['filter']['amount']:return f"滤镜浓淡 {f['amount']*100:g}%"
        names={'saturation':'饱和度','hue':'色相','key':'影调','contrast':'对比度',
               'highlight':'高光对比度','shadow':'阴影对比度','sharpness':'锐度',
               'filter_effect':'黑白滤镜','mono_toning':'黑白色调','toning':'色调','tone':'色调'}
        changes=[(key,x) for key,x in f['controls'].items() if before['filter']['controls'].get(key)!=x]
        if len(changes)==1:
            key,x=changes[0];return '滤镜'+names.get(key,key)+' '+value(x)
        return '滤镜参数'
    if before['denoise']!=after['denoise']:
        d=after['denoise'];old=before['denoise']
        if d['kind']==old['kind']=='native':
            from native_controls import NR_CONTROLS
            changes=[(k,x) for k,x in d['native'].items() if old['native'].get(k)!=x]
            if len(changes)==1:
                key,x=changes[0];return NR_CONTROLS[key]['name_zh']+' '+str(x)
        if d['kind']==old['kind']=='ai' and d['model']==old['model'] and d['amount']!=old['amount']:
            return f"降噪结果混合 {d['amount']:g}%"
        return {'off':'关闭降噪','native':'原厂降噪','ai':'AI 降噪 '+{'nafnet':'NAFNet','restormer':'Restormer','scunet':'SCUNet'}[d['model']]}[d['kind']]
    if before['exposure']!=after['exposure']:return f"曝光 {after['exposure']:+.2f} EV"
    if before['balance']!=after['balance']:return f"色彩平衡 X {after['balance'][0]:+g}, Y {after['balance'][1]:+g}"
    prior,current=before['post'],after['post']
    if prior['tone']['regions']!=current['tone']['regions']:
        changes=[i for i,(a,b) in enumerate(zip(prior['tone']['regions'],current['tone']['regions'])) if a!=b]
        if len(changes)==1:
            i=changes[0];return ('黑色','阴影','中间调','高光','白色')[i]+' '+value(round(current['tone']['regions'][i]*50,6))
    if prior['tone']['points']!=current['tone']['points']:return '曲线调整'
    colours={'red':'红色','orange':'橙色','yellow':'黄色','green':'绿色','aqua':'青色','blue':'蓝色','purple':'紫色','magenta':'洋红色'}
    mixer_changes=[(colour,key,x) for colour,values in current['mixer'].items() for key,x in values.items() if prior['mixer'][colour][key]!=x]
    if len(mixer_changes)==1:
        colour,key,x=mixer_changes[0]
        return colours[colour]+{'hue':'色相','saturation':'饱和度','luminance':'明亮度'}[key]+' '+value(x)
    def diff(a,b):
        for key,value in b.items():
            if a.get(key)!=value:
                if isinstance(value,dict):yield from diff(a.get(key,{}),value)
                else:yield key,value
    changes=list(diff(before['post'],after['post']))
    if len(changes)==1:
        key,value=changes[0];name=_LABELS.get(key,key)
        return name+' '+(f'{value:g}' if type(value) in (int,float) else '调整')
    return '调色调整'
