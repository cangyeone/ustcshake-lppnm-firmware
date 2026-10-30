# Attribution and license scope

The new firmware source, portable incremental runtime, conversion tools, and the generated LPPNM artifacts distributed **in this repository** are provided under the root MIT license. This does not relicense third-party libraries, fonts, the original manufacturer's firmware, or the entire upstream pnsn repository.

## LPPNM

Model architecture/checkpoint origin: [cangyeone/pnsn](https://github.com/cangyeone/pnsn), commit `1f955f4d8cffbc5c43584f02ef14a0da2b0aeb6b`, `models/LPPNM.py` and `ckpt/china.lppnm.pt`. The generated fused/quantized model header and parameter archive retain that provenance. The original checkpoint and raw calibration waveforms are not copied into this repository. See `docs/validation.json` for the checkpoint hash and numerical validation.

The upstream repository had no explicit repository-wide license declaration when inspected on 2026-10-30. The MIT grant here covers the firmware and generated model files published here by the same repository owner; it is not a statement about the licensing of other upstream materials.

## U8g2 and fonts

U8g2 2.36.17 is downloaded by PlatformIO, not vendored as source. Its graphics code is BSD-2-Clause; fonts have separate licenses. A copy of the upstream license is in `LICENSES/U8g2.txt`.

- `u8g2_font_wqy12_t_gb2312`: WenQuanYi Bitmap Song, GPL v2 with font embedding exception. Copyright belongs to the WenQuanYi project and contributors. The original BDF header is retained in `LICENSES/WenQuanYi-font-header.txt`. [Font source and licensing](https://github.com/olikraus/u8g2/wiki/fntgrpwqy), [BDF source](https://github.com/larryli/u8g2_wqy/blob/master/bdf/wenquanyi_9pt.bdf).
- `LICENSES/WenQuanYi-conversion.txt` is the MIT license of Larry Li's conversion repository; it does not replace the original font license.
- `u8g2_font_5x7_tf`: the upstream generated source identifies this as a public-domain Misc Fixed bitmap font.
- The small raster glyphs in `include/paper.hpp` are implemented within this project.

## Arduino / Espressif

Arduino ESP32 core 2.0.17 is LGPL-2.1-or-later; see `LICENSES/Arduino-ESP32.txt` and [corresponding source](https://github.com/espressif/arduino-esp32/tree/2.0.17). The bundled Espressif SDK and components retain their individual notices, including Apache-2.0 and other licenses. PlatformIO resolves these dependencies; this repository includes complete application sources so users can rebuild and relink them.

## Device manufacturer

The original full-flash images, NVS, binary disassembly, and manufacturer logos are not redistributed. Hardware pin mappings and protocol constants describe interoperability with the tested board. This project is an independent firmware rewrite and does not imply manufacturer endorsement.
