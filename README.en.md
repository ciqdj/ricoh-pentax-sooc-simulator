# RICOH & PENTAX SOOC Simulator

English · [Simplified Chinese](README.md)

![RICOH & PENTAX SOOC Simulator](src/branding/startup-cover.png)

**Use Pentax's original color processing on photos from Canon, Sony, Nikon, and Fujifilm.**

Open a RAW or a finished photo and try Pentax's four seasonal looks, Reversal Film, or Ricoh GR IV Cross Processing. The project makes those color choices available in one Windows editor. It grew out of work on Digital Camera Utility 5, the Pentax K-3 Mark III firmware, and community research on the Ricoh GR IV.

[Windows downloads](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/latest) · [Pipeline](docs/PIPELINE.md) · [Build instructions](docs/BUILD.md) · [Disclaimer](DISCLAIMER.md)

Version 3.5.12 adds a language menu in the top bar. Select **English** to change the interface. The app remembers your selection. The 3.5.11 public Windows download uses the earlier interface.

## What it does

**Pentax.** The workbench calls the original color processing through an adapter for Digital Camera Utility 5. It supports 17 modes, including HARUBENI, KATEN, KYUSHU, and FUYUNO. Mode controls use the available factory parameters for tone, contrast, saturation, monochrome filtering, and other adjustments. Pentax PEF can use the native RAW path. Other brands' RAW files go through LibRaw and enter the engine as 16-bit RGB.

**Ricoh GR IV.** The R3 implementation covers 11 modes and 17 tone combinations. It uses recovered mode matrices, tone curves, and saturation parameters alongside work from GRKit and other community projects. Linear RAW, ordinary sRGB, and Ricoh Standard output have separate input paths. Contrast, highlights, shadows, and saturation update the preview and export using the same calculations.

**Editing and export.** Compare before and after, inspect details at full size, and keep each photo's edits when switching looks. Import CUBE LUTs, use the optional NAFNet or Restormer input denoiser, and export full-size images. PNG and TIFF support 16-bit lossless output. The original photos are opened read-only.

## Download the looks and research data

| Download | What you get |
| --- | --- |
| [Pentax CUBE and XMP](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/download/v3.5.11/PENTAX-Filters-3.5.11.zip) | 17 sampled native looks, including the four seasons, and 17 creative looks. |
| [Ricoh GR IV CUBE and XMP](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/download/v3.5.11/RICOH-GR-IV-Filters-3.5.11.zip) | 11 modes and 17 tone combinations sampled from the R3 color model. |
| [Recovered color data](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/releases/download/v3.5.11/SOOC-Recovered-Color-Data-3.5.11.zip) | DCU curves and matrices, K-3 Mark III mode records, seasonal parameters, and the runtime tables. |

Use the CUBE files in an sRGB workflow or import the XMP profiles into Lightroom Classic. These downloads store fixed 33-point color grids. The app handles RAW development, denoising, and spatial adjustments. See the [file index and import notes](docs/FILTERS.md) for details.

## Testing and open questions

The Windows tests cover RAW input routing, color controls, and CPU and OpenCL output on reference inputs. The project still needs broader camera samples and same-RAW, same-settings comparisons against the original applications and cameras. The GR IV implementation covers the recovered color stages, with hardware development and three additional modes still under investigation. Windows 10 and 11 x64 are the target platforms.

Want to help? A report with the camera model, app version, settings, and a sample you can share makes a color difference much easier to trace. See the [verification records](docs/verification) and [open issues](https://github.com/xxhgkd/ricoh-pentax-sooc-simulator/issues).

## Credits and licensing

Thanks to [GRKit](https://github.com/ahonn/GRKit), [Ricoh-GRMod](https://github.com/Maxwooow/Ricoh-GRMod), and [gr3-film-mods](https://github.com/idontknownihil/gr3-film-mods) for publishing their research. The denoising options use [NAFNet](https://github.com/megvii-research/NAFNet) and [Restormer](https://github.com/swz30/Restormer).

Project-written code is available under the [MIT license](LICENSE). Manufacturer components and data, third-party models and LUTs, and brand assets retain their own terms. This is an independent community project. AI coding agents assisted with parts of the analysis, software, and documentation. See the [third-party notices](THIRD_PARTY.md) and [disclaimer](DISCLAIMER.md).
