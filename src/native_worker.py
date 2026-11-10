"""One disposable process per original-engine render; no fitted colour code."""
from pathlib import Path
import ctypes as C, json, os, struct, sys, time
from native_host import Host,emit
from native_controls import initialize as initialize_controls


def run(request_file):
    request_file=Path(request_file).resolve()
    request=json.loads(request_file.read_text(encoding='utf-8'))
    working=request_file.parent
    def stage(text):
        (working/'stage.json').write_text(json.dumps({'stage':text}),encoding='utf-8')
    stage('加载原厂引擎')
    os.environ['TEMP']=os.environ['TMP']=str(working)
    source=Path(request['input']).resolve();destination=Path(request['output']).resolve()
    if not source.is_file() or destination.exists() or source==destination:
        raise ValueError('Invalid render paths')
    mode=int(request['mode']);ev=float(request.get('exposure_ev',0.0))
    if mode not in {-1,0,1,16,17,18,19,20,23,24,25,27,28,29,30,31,32,33} or not -5<=ev<=5:
        raise ValueError('Unsupported native setting')
    started=time.monotonic();host=Host()
    assert host.call(0x13a2634,C.c_int,[C.c_int],1)
    assert host.call(0x13a25b0,C.c_int,[])
    host.initialize_full_crt()
    status=host.call(0xae3fb0,C.c_int16,
        [C.c_byte,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,
         C.c_byte,C.c_byte,C.c_byte,C.c_byte,C.c_void_p,C.c_void_p],
        1,0,0,0,2,0,1,1,1,1,0,0)
    if status:raise RuntimeError(f'Native initialization failed ({status})')
    stage('读取原厂参数')
    emit('native_string_codepages',narrow=C.c_int.from_address(host.base+0x1c45420).value,
         filesystem=C.c_int.from_address(host.base+0x1c43834).value)
    keep=[]
    def path_object(path):
        storage=C.create_string_buffer(0x1000);keep.append(storage);obj=C.addressof(storage)
        host.call(0xe07be0,C.c_void_p,[C.c_void_p],obj)
        # The original GUI passes UTF-16 CString data to this path setter.
        # Its inspected implementation only reads length and character data;
        # it does not retain, release or mutate this caller-owned string view.
        encoded=str(path).encode('utf-16le');units=len(encoded)//2
        if units>=510:raise ValueError('文件路径过长，请移到较短的文件夹路径')
        wide=C.create_string_buffer(24+len(encoded)+2);keep.append(wide)
        struct.pack_into('<iii',wide,8,units,units,1)
        C.memmove(C.addressof(wide)+24,encoded,len(encoded))
        pointer=C.c_void_p(C.addressof(wide)+24);keep.append(pointer)
        host.call(0xe0acb0,None,[C.c_void_p,C.c_void_p],obj,C.addressof(pointer))
        return obj
    native_path=path_object(source);target_path=path_object(destination)
    ui_storage=C.create_string_buffer(0x1000);ui=C.addressof(ui_storage)
    host.call(0xa995d0,C.c_void_p,[C.c_void_p,C.c_void_p],ui,0)
    # Run the original mode selection/default loading path. A mode number alone
    # omits the mode's contrast, key, sharpness and additional colour settings.
    controls,extra=initialize_controls(host,source,ui,keep,mode,ev,
                                       request.get('controls'),factory=mode!=-1)
    # Use the original upper-level dispatcher targets. RGB is converted by
    # acab40 inside adcbd0; PEF is converted by aaaee0 inside abf2a0. These
    # paths also retain aaf100 and their original image-transfer behaviour.
    # Bypass only the GUI's camera eligibility check by calling the target.
    # 0x40 is the original 64-bit/pixel working format (RGB16 plus padding),
    # selected by 442210 for TIFF16 (format 0x31). -2 asks the image dispatcher
    # for its default display format and silently quantizes to 8 bits.
    stage('原厂引擎处理照片')
    if source.suffix.lower()=='.pef':
        sdk_entry='abf2a0'
        stream=host.call(0xabf2a0,C.c_void_p,
            [C.c_void_p,C.c_int,C.c_byte,C.c_int,C.c_byte,C.c_void_p,C.c_byte,
             C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p],
            ui,0,1,0x40,1,0,0,native_path,0,0,0,0,host.base+0xa9d190,0)
        extra_gamma=1
    else:
        sdk_entry='adcbd0'
        stream=host.call(0xadcbd0,C.c_void_p,
            [C.c_void_p,C.c_int,C.c_byte,C.c_int,C.c_byte,C.c_void_p,
             C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p],
            ui,0,1,0x40,1,0,native_path,0,0,0,host.base+0xa9d190,0)
        extra_gamma=0
    if not stream:raise RuntimeError('原厂处理入口未生成图像')
    if controls['additional_controls_active']:
        stage('原厂色调处理')
        stream=host.call(0xab14e0,C.c_void_p,[C.c_void_p,C.c_void_p,C.c_byte],stream,extra,extra_gamma)
        if not stream:raise RuntimeError('原厂色调处理未生成结果')
    stage('生成 16 位处理结果')
    status=host.call(0x43d7c0,C.c_int16,
        [C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
         C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,
         C.c_int,C.c_void_p,C.c_void_p,C.c_byte,C.c_void_p,C.c_byte,C.c_int],
        target_path,stream,0,0x31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    if status:raise RuntimeError(f'Native TIFF writer failed ({status})')
    if host.blocked_ui:raise RuntimeError('Unexpected native UI request')
    result={'original_native_code':True,'native_mode':controls['native_mode'],'exposure_ev':ev,
            'full_resolution':True,'native_bits':16,'gui_entry_called':False,
            'post_render_colour_adjustment':False,'oracle_equivalence_verified':False,
            'seconds':round(time.monotonic()-started,3)}
    result['original_sdk_entry']=sdk_entry
    result['native_working_format']=0x40
    result['factory_defaults']=controls['factory_defaults']
    result['effective_controls']=controls['effective_controls']
    result['factory_noise_reduction']=controls['factory_noise_reduction']
    result['effective_noise_reduction']=controls['effective_noise_reduction']
    result['original_gui_default_loader']=True
    result['additional_controls_active']=controls['additional_controls_active']
    result['available_controls']=controls['available_controls']
    result['highlight_shadow_enabled']=controls['highlight_shadow_enabled']
    result['factory_highlight_shadow_enabled']=controls['factory_highlight_shadow_enabled']
    result['original_additional_colour_stage']=bool(controls['additional_controls_active'])
    (working/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main(request_file):
    request_file=Path(request_file)
    with (request_file.parent/'worker.log').open('w',encoding='utf-8') as log:
        sys.stdout=sys.stderr=log
        try:run(request_file)
        except Exception:
            import traceback
            traceback.print_exc();return 1
    return 0

if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
