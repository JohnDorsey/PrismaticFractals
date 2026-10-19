#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 John Dorsey. See the file LICENSE.
"""Prismatic Fractals: run this file (python PrismaticFractals.py).

The program is the `PrismaticFractals` package next to this launcher; its docstring (PrismaticFractals/__init__.py) is the
description of what the program does. This file exists so that `python PrismaticFractals.py` keeps working and so that the
names of the files the program writes follow THIS file's name (rename it and they change).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import PrismaticFractals

# The names of the files the program writes follow this file's name.
PrismaticFractals.paths.PROGRAM_NAME = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

# This file exposes every name of the package too, as the launcher always has. (It has the package's name, so a script
# in this folder that says `import PrismaticFractals` gets the package itself - the folder with the same name wins over this file.)
globals().update({_name: _value for _name, _value in vars(PrismaticFractals).items()
                  if not _name.startswith("__")})

if __name__ == "__main__":
    main()
