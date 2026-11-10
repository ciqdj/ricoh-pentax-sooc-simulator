# 滤镜文件与原厂数据

| 下载 | 内容 |
| --- | --- |
| [宾得 CUBE / XMP](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/download/v3.5.11/PENTAX-Filters-3.5.11.zip) | 34 套。17 套由 DCU 5.10.2.2 使用 K-1 参数采样，含四季。另有 17 套 3.1 创意组合、黑白变体和基调曲线。 |
| [理光 GR IV CUBE / XMP](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/download/v3.5.11/RICOH-GR-IV-Filters-3.5.11.zip) | R3 色彩模型的 11 种模式、17 组色调。电影黄调、电影绿调和交叉处理各有三档。 |
| [原厂色彩拆解数据](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/download/v3.5.11/SOOC-Recovered-Color-Data-3.5.11.zip) | DCU 曲线和矩阵、固件模式记录、四季参数，以及宾得和理光的运行数据。 |

Photoshop 中先把照片转换到 sRGB，再用「颜色查找」调整图层载入 CUBE。Lightroom Classic 中导入 XMP，再从配置文件浏览器选择。两种格式都保存 33³ 网格上的固定色彩映射。全部文件已解码核对，Adobe 宿主内的实际加载验证待补。

## 拆解文件

- [DCU 曲线](../data/pentax-recovered/dcu-curves)保存 GOLD 相关分量与交叉处理共 6 条曲线。每条 4096 点，提供小端 U16、CSV 和 JSON。
- [DCU 静态参数](../data/pentax-recovered/dcu-parameters)记录 124 条回退矩阵，以及黑白矩阵和对比度参数的来源。
- [K-3 III 固件模式记录](../data/pentax-recovered/k3iii-firmware/firmware_mode_records.json)取自 2.20 版固件，共 45 条，包含四季。[官方固件页](https://www.ricoh-imaging.co.jp/english/support/digital/k-3-3.html)
- [K-3 III PEF 参数](../data/pentax-recovered/k3iii-pef)包含 34 组矩阵与两条基础曲线。
- [四季参数](../data/pentax-recovered/seasonal-banks/index.json)包含从公开样片的相机参数标签中读取的 10 组记录。匹配到的条目附公开来源链接。
- [宾得运行参数](../src/native-data)、[理光 GR IV 模式矩阵、曲线和动态参数](../src/gr4-data)也在数据包中。

[校验清单](../data/pentax-recovered/manifest.json) · [原厂处理流程](PIPELINE.md) · [CUBE / XMP 导出代码](../src/filter_export.py) · [富士模拟和经典胶片 LUT](../src/builtin-luts) · [来源与许可](../THIRD_PARTY.md)
