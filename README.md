# RICOH & PENTAX SOOC Simulator

![RICOH & PENTAX](src/branding/startup-cover.png)

**让佳能、索尼、尼康、富士的照片也能用上宾得原厂调色引擎。**

我们拆解了宾得官方软件 **Digital Camera Utility 5（DCU 5）** 和 **K-3 III 官方固件更新**。从 DCU 5 拆出原厂色彩处理链的入口，让原厂引擎独立运行，同时从固件里提取了四季等模式的原厂记录。**现在佳能、索尼、尼康、富士的 RAW 和普通照片，也能用这套引擎跑 17 种宾得原厂模式。**

嫌自己的照片偏色、发灰、太寡淡，就试试宾得四季、反转片，或者换到理光 GR IV 的交叉处理。跨品牌输入适配、原厂参数调用和工作台代码都已开源，想研究宾得色彩的朋友可以接着往下拆。

[下载 Windows 版](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/latest) · [处理流程与代码入口](docs/PIPELINE.md) · [源码运行与打包](docs/BUILD.md)

[宾得滤镜包](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/download/v3.5.11/PENTAX-Filters-3.5.11.zip) · [理光 GR IV 滤镜包](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/download/v3.5.11/RICOH-GR-IV-Filters-3.5.11.zip) · [原厂拆解数据](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/download/v3.5.11/SOOC-Recovered-Color-Data-3.5.11.zip) · [文件清单与导入方法](docs/FILTERS.md)

## 我们做了什么

### 把宾得原厂处理接到跨品牌照片上

这里最有意思的工作，是把 DCU 原来和桌面程序绑在一起的处理入口接出来，让它可以在独立后台进程里运行。程序加载指定版本的 DCU，建立它需要的参数对象，调用原有的默认设置、调参和图像处理函数。原厂安装文件保持原样。

- **17 种原厂模式**：Natural、Bright、Portrait、Landscape、MIYABI、Monochrome、Muted、Reversal、Bleach Bypass、Radiant、Flat、SATOBI、HARUBENI、KATEN、KYUSHU、FUYUNO、GOLD。
- **按模式开放参数**：饱和度、色相、明暗调、对比度、高光、阴影、锐度、黑白滤色等，遵循已确认的 DCU 参数调用分支。
- **四季参数接入**：程序按输入机型选择 K-1、K-1 II 或 K-3 III 参数银行，跨品牌 RGB 使用 K-1 参数银行。
- **两条输入路径**：未修改的 PEF 全尺寸处理走原厂 RAW 入口；其他品牌 RAW 先用 LibRaw 解码，再通过 RGB16 兼容输入进入 DCU。JPEG、PNG、TIFF 等也可以用。
- **把处理放到后台**：界面、原厂计算、AI 降噪分开；原厂进程出错时结束当前任务，主窗口保留。

这几部分的实现都在 `src/native_*.py`。我们开源的是原厂引擎的调用与适配工作；DCU 的二进制文件和内部代码版权仍归原权利人。

### 理光 GR IV：接入已有研究，补上动态参数

理光部分沿用 GRKit、GR Mod 等项目公开的研究，并对照 GR IV 固件中的函数和参数。感谢这些作者把入口、结构和推导过程公开出来。

3.5.11 补了三件之前欠着的事：

1. **分清输入**。线性 RAW、普通 sRGB 照片、已经经过理光 Standard 处理的成片，各走对应的入口。其他品牌的普通照片直接按 sRGB 解码。
2. **让影调参数真正改曲线**。明暗调、对比度、高光对比度和阴影对比度接入原厂整数曲线生成算法，预览和导出共用这套计算。
3. **让模式参数可以单独调**。饱和度使用原厂生成的后级矩阵；黑白模式隐藏饱和度，电影色调和交叉处理保留各自档位。

目前有 11 种理光模式、17 组默认色调组合。矩阵、分色相、曲线、后级矩阵和显示处理在 CPU 与 GPU 路径中保持一致。GR IV 的完整硬件显影链仍在研究，当前版本覆盖已经实现的色彩阶段。

### 平时修照片需要的工作台

- 照片调色按照片保存，换滤镜保留曝光、曲线、影调和颜色调整。
- 历史记录、可折叠调整区、鼠标位置缩放、拖动平移、前后分界对比。
- 左侧缩略图用于看风格，调色和降噪完成时保持稳定。
- 自建品牌、分类，导入 1D / 3D CUBE；内置 Fuji 第三方模拟与精选 20 款经典胶片 LUT，附来源和许可。
- DCU 原厂降噪，以及独立输入阶段的 NAFNet、Restormer。AI 完成后复用工作副本，换滤镜继续用。
- 启动后后台预热上次使用的 AI 模型，首次默认 NAFNet；设备测速与模型会话留在常驻进程里。顶部任务区显示阶段，也可以取消预热。
- JPEG、PNG、TIFF、WebP、BMP 输出；PNG / TIFF 可用 16 位无损编码。原文件只读，导出移除作者、GPS、序列号等个人元数据。
- 导出 Photoshop CUBE 和 Lightroom Classic XMP 配置文件。LUT 保存固定色彩映射，空间降噪和完整 RAW 显影继续由应用内处理链负责。

## 这版检查到了哪里

这次检查重点放在改动经过的实际路径，记录放在 [docs/verification](docs/verification)。

- 17 组默认理光设置的曲线、矩阵与旧数据一致，原 Standard 输入路径的像素结果保留。
- 新输入路径、动态参数、曝光和浓淡经过 Python / 原生 CPU / OpenCL 对照，测试样本的 RGB16 输出逐像素一致。
- 原厂曲线生成入口另做了 4 组边界及组合参数检查，与 ARM 执行结果逐点相等。
- 一张 5796 × 3870 的佳能 CR2 检查了线性输入、手动调色后的输入切换、解码复用及换滤镜保留调整。
- 隐藏窗口中的真实参数控件检查了多个调整共存、模式切换和黑白参数范围。
- 在 RTX 3060 Laptop GPU 这台测试机上，NAFNet 首次后台准备约 46 秒；同一进程再次准备同一模型约 0.016 秒。整张照片的推理时间另算。

## 目前的边界

宾得路径确实在执行原厂模块。跨品牌 RAW 的解码和 RGB 输入适配由我们负责；全机型、全模式、全部参数的同图同设置 DCU 像素对比还需要补齐。原厂降噪在跨品牌 RGB 输入上的噪声校准也需要继续核对。

理光目前开放 11 种模式。漂白旁路、复古、HDR 及完整机内硬件处理尚未接通，也还缺 GR IV 真机的同一 RAW、多模式开发结果作为对照。

AI 降噪作用于解码后的 RGB16 工作副本。它保留原 RAW，但产物是处理后的像素图；传感器马赛克、相机校准和拍摄信息不会因此变成另一台相机的数据。

Windows 10 / 11 x64 是当前交付平台。显卡通过精度检查和实测后启用，设备不适用时走 CPU。不同 RAW 格式的解码范围随 LibRaw 版本变化，佳能 CR2 之外的相机组合还需要更多样片验证。视频入口目前沿用旧适配路径，本轮新增理光参数面向照片。

## 接下来想一起做的事

最想补的是宾得同图同设置的原版对照集，把各模式和参数组合逐个记清楚。随后继续补理光的缺失模式、真实相机运行数据，以及跨品牌输入的色彩管理。降噪会继续做设备选择、模型复用和后台调度，优先保留画质与稳定性。

欢迎带着具体相机型号、软件版本、参数和可分享的样片来提 Issue，也欢迎直接看代码、提 PR。遇到效果差异时，把产生差异的那一步找出来，会比一句“看起来不像”更容易往前推进。

## 感谢与许可

- [GRKit](https://github.com/ahonn/GRKit)：输入域、预览色彩处理与原厂整数影调算法；本次对照版本 `6ab9a2522e6e3103f62d825e55beb2630f45d740`。
- [Ricoh-GRMod](https://github.com/Maxwooow/Ricoh-GRMod)、[gr3-film-mods](https://github.com/idontknownihil/gr3-film-mods)：处理入口、参数结构与模式研究。
- [NAFNet](https://github.com/megvii-research/NAFNet)、[Restormer](https://github.com/swz30/Restormer)：降噪模型。
- [FujifilmCameraProfiles](https://github.com/abpy/FujifilmCameraProfiles)、[Film-Luts](https://github.com/YahiaAngelo/Film-Luts)：第三方 LUT，具体来源和许可随资源保留。

项目自己编写的代码采用 [MIT 许可](LICENSE)。第三方代码、模型、滤镜数据和品牌素材按各自许可使用，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。这是独立社区项目，RICOH、PENTAX、GR 等名称与标志属于各自品牌。
