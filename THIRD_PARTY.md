# 资源来源与许可

本仓库的 MIT 许可用于项目自己编写的代码。以下资源保留各自的来源、版权与许可。

| 内容 | 来源与处理方式 | 许可记录 |
| --- | --- | --- |
| 理光色彩数学和整数影调算法 | GRKit；Python、C 和 OpenCL 适配 | `src/gr4-data/LICENSE-GRKit.txt`，MIT |
| 宾得原厂执行模块 | Digital Camera Utility 5.10.2.2；运行时加载原厂程序 | 原权利人保留版权；Git 源码树中未放入 DCU EXE / DLL |
| 宾得机型参数、理光模式表 | 从指定原厂软件 / 固件取得的参数数据 | 来源记录随数据保存；项目 MIT 许可不重授权原厂内容 |
| NAFNet / Restormer | 官方推理模型，FP32 ONNX 格式 | 运行包 `denoise-data/LICENSE-*.txt` 与 manifest |
| Fuji 第三方 LUT | abpy/FujifilmCameraProfiles | CC BY-NC-SA 4.0，见 `src/builtin-luts/sources/` |
| 精选胶片 LUT | YahiaAngelo/Film-Luts | 对应源仓库许可，见 `src/builtin-luts/sources/` |
| LibRaw / rawpy、ONNX Runtime、FFmpeg、Python、NumPy、Pillow 等 | 图像解码、模型运行、视频与应用运行环境 | `src/licenses/` 和运行包中的许可证 |
| 原生计算内核的编译运行库 | Zig / MinGW | `src/gr4-data/MinGW-COPYING.txt` |
| RICOH / GR / PENTAX 名称及标志 | 品牌识别与程序图标 | 归各自权利人；项目与品牌无隶属关系 |

分发第三方 LUT 时请保留资源自带的许可，尤其注意 Fuji 这组的非商业条款。替换模型或 LUT 时也应更新来源记录。
