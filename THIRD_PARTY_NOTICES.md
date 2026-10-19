# Third-party notices

PrismaticFractals is Copyright (C) 2026 John Dorsey and is released under the GNU General Public License, version 3 or (at your option) any later version; see `LICENSE`.

This repository contains no third-party source code or binaries. The software below is installed separately by whoever runs the program; it is listed here so that its licenses and authors are acknowledged, and so that it is clear which of them are only run as separate programs.

## Python packages (imported by the program)

| Package | Used for | License | Tested with |
|---|---|---|---|
| [numpy](https://numpy.org) | arrays everywhere | BSD-3-Clause (its metadata lists BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 for the bundled parts) | 2.4.4 |
| [numba](https://numba.pydata.org) | compiles the simulation kernels | BSD-2-Clause (its metadata says "BSD") | 0.67.0 |
| [llvmlite](https://github.com/numba/llvmlite) | installed with numba | BSD-2-Clause (not stated in its installed metadata) | 0.49.0 |
| [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) | builds the command line for ffmpeg | Apache-2.0 (its metadata: "Apache Software License") | 0.2.0 |
| [Pillow](https://python-pillow.org) | optional: development screenshots only | MIT-CMU (HPND) | 12.2.0 |

The Apache-2.0 license of ffmpeg-python is compatible with GPL-3.0 (it is not compatible with GPL-2.0-only).

## Programs the user installs and the program runs as separate processes

These are not distributed with this repository and are not linked into it, so their licenses do not apply to this program's own code.

| Program | Used for | License |
|---|---|---|
| [FFmpeg](https://ffmpeg.org) | encoding video, and JPEG XL stills | LGPL-2.1-or-later, or GPL-2.0-or-later / GPL-3.0 for builds that enable GPL parts (the build this was developed with, from gyan.dev, 9.0.2, is a GPL-3.0 build) |
| `cjxl` from [libjxl](https://github.com/libjxl/libjxl) | animated JPEG XL, and stills above 2^28 pixels | BSD-3-Clause |
| Python with tkinter | the interpreter and the user interface | Python Software Foundation License |

If you redistribute FFmpeg or `cjxl` together with this program, you must follow their licenses for those files.

## Development tools (not part of the program)

`dev/tools/wallcrop.nim` is a standalone tool that needs the [Nim](https://nim-lang.org) compiler (MIT), the programs of [libvips](https://www.libvips.org) (LGPL-2.1-or-later; run as separate processes) and the [SDL2](https://libsdl.org) library (zlib license; loaded at run time). None of them is distributed here.

## Credits for techniques and formats

- The progress bar in the Video tab that shows the frame count as text on the bar (`make_labeled_progress_style` in `PrismaticFractals/app_video.py`) uses a technique taken from an answer at <https://stackoverflow.com/questions/74895125/how-to-customize-tkinter-ttk-progressbar-with-label-text-and-custom-bar-color>. Stack Overflow content from this period is licensed CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>), which allows it to be adapted into a project under GPL-3.0 when credited, as it is here.
- The QOI image format, which the program can write, was designed by Dominic Szablewski (<https://qoiformat.org>; its reference implementation is MIT-licensed). The encoder in `PrismaticFractals/image_io.py` implements that format.
- The equation of the Hügelschäffer egg shape was transcribed from the page on <https://www.mathcurve.com>.
