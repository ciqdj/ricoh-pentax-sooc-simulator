"""Keep native workers free of the GUI, NumPy and legacy video import costs."""
import sys

def main():
    checks={'--verify-latest-ui':'verify_latest_ui','--verify-live-post':'verify_live_post',
            '--verify-gr4-controls':'verify_gr4_controls',
            '--verify-spatial':'verify_spatial_inspection','--verify-native-noise-ui':'verify_native_noise_ui',
            '--verify-fine-controls':'verify_ui_controls','--verify-basic-export':'verify_basic_export',
            '--verify-neural-host':'verify_neural_process','--verify-ai-models':'verify_ai_models',
            '--verify-async-close':'verify_async_close','--verify-panel-layout':'verify_panel_layout',
            '--verify-control-lifecycle':'verify_control_lifecycle',
            '--verify-denoise-startup':'verify_denoise_startup',
            '--verify-photo-denoise':'verify_photo_denoise',
            '--verify-source-denoise-ui':'verify_source_denoise_ui',
            '--verify-adjustment-live':'verify_adjustment_live',
            '--verify-custom-lut':'verify_custom_lut',
            '--verify-builtin-luts':'verify_builtin_luts',
            '--verify-custom-lut-ui':'verify_custom_lut_ui',
            '--verify-preset-cache':'verify_preset_cache',
            '--verify-neural-devices':'verify_neural_devices',
            '--verify-photo-edit':'verify_photo_edit',
            '--verify-editor-pipeline':'verify_editor_pipeline',
            '--verify-editor-workflow':'verify_editor_workflow',
            '--verify-editor-custom':'verify_editor_custom',
            '--verify-editor-visual':'verify_editor_visual',
            '--verify-editor-denoise-detail':'verify_editor_denoise_detail',
            '--verify-photo-task-queue':'verify_photo_task_queue',
            '--verify-release-batch':'verify_release_batch'}
    if '--verify-photo-cache' in sys.argv:
        from verify_photo_denoise import main as verify_cache
        verify_cache(sys.argv[sys.argv.index('--verify-photo-cache')+1],real_model=False)
        raise SystemExit(0)
    for flag,module in checks.items():
        if flag in sys.argv:
            import importlib
            importlib.import_module(module).main(sys.argv[sys.argv.index(flag)+1])
            raise SystemExit(0)
    if '--verify-native-noise-comparison' in sys.argv:
        from verify_native_noise_comparison import verify
        verify(sys.argv[sys.argv.index('--verify-native-noise-comparison')+1])
        raise SystemExit(0)
    if '--ai-denoise-worker' in sys.argv:
        from neural_process import worker_main
        raise SystemExit(worker_main())
    if '--verify-detail-352' in sys.argv:
        from verify_352_detail import main
        main(sys.argv[sys.argv.index('--verify-detail-352')+1])
        raise SystemExit(0)
    if '--verify-viewport-live' in sys.argv:
        from verify_viewport_live import main
        main(sys.argv[sys.argv.index('--verify-viewport-live')+1])
        raise SystemExit(0)
    if '--verify-351' in sys.argv:
        from verify_351 import main
        main(sys.argv[sys.argv.index('--verify-351')+1])
        raise SystemExit(0)
    if '--native-worker' in sys.argv:
        from native_worker import main
        raise SystemExit(main(sys.argv[sys.argv.index('--native-worker')+1]))
    if '--dcu-denoise-worker' in sys.argv:
        from dcu_denoise import worker_main
        raise SystemExit(worker_main(sys.argv[sys.argv.index('--dcu-denoise-worker')+1]))
    try:
        from workbench_app import main
        main()
    except Exception:
        import traceback
        from app_paths import report_dir
        (report_dir()/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        if not any(x.startswith('--') for x in sys.argv[1:]):
            from tkinter import messagebox
            messagebox.showerror('启动失败','启动失败，详情已写入本机日志。')
        sys.exit(1)

if __name__=='__main__':
    try:main()
    except Exception:
        # Command-line verification/worker failures must remain headless even
        # before the GUI dispatcher imports, including windowed frozen builds.
        import traceback
        from app_paths import report_dir
        (report_dir()/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        if not any(x.startswith('--') for x in sys.argv[1:]):
            from tkinter import messagebox
            messagebox.showerror('启动失败','启动失败，详情已写入本机日志。')
        sys.exit(1)
