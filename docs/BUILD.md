# 源码运行与打包

当前环境：Windows 10 / 11 x64、Python 3.12。工作目录、虚拟环境、构建目录可放在空间充足的数据盘。

## 准备

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Git 仓库包含应用代码、参数资源、LUT 与许可证。较大的 AI 模型、运行库、视频工具和兼容引擎文件在 Windows 发布包中。先下载相同版本的 EXE，再从中提取运行资源：

```powershell
.venv\Scripts\python tools\import_runtime.py "D:\Downloads\RICOH-PENTAX-SOOC-Simulator-3.5.11.exe"
.venv\Scripts\python src\workbench_entry.py
```

提取器只接受资源白名单，写入 `src`；不会从发布包替换 Python 源文件。宾得宿主锁定 DCU 5.10.2.2，版本校验在 `src/native_host.py` 中。源码模式也可使用本机对应版本的 DCU 安装。

## 运行针对本次改动的检查

```powershell
.venv\Scripts\python src\verify_gr4_controls.py build\verification
```

检查包括旧默认数据、新输入编码、动态参数、CPU / GPU 输出和隐藏窗口中的真实参数控件。没有可用显卡时仍可完成 CPU 检查。原厂同图对照需要另备对应版本 DCU 和样片。

## 生成 EXE

```powershell
.venv\Scripts\python tools\package.py --work build --output dist
```

这个命令生成 PyInstaller 单文件内包，包含已有运行资源。仓库也提供了原生启动器的 C 源码：`portable_launcher.c`、`launcher_splash.c`。正式发布包在内包外加上这个启动器，显示启动横幅并提前选择临时目录。

修改原生理光内核时，用 Zig 0.17.1 或兼容的 Windows x64 C 编译器重新编译 `src/gr4_core.c`，保留 `-ffp-contract=off -fno-fast-math`。替换 DLL / OpenCL 源码后，须同步 `gr4-data/manifest.json` 的组件摘要，并重新运行检查。
