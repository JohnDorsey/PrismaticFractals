#!/usr/bin/env python3
"""Light bounces: run this file (python light_bounces.py).

The program is the `lightbounces` package next to this launcher; its docstring (lightbounces/__init__.py) is the
description of what the program does. This file exists so that `python light_bounces.py` keeps working and so that the
names of the files the program writes follow THIS file's name (rename it and they change).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightbounces

# The names of the files the program writes follow this file's name.
lightbounces.paths.PROGRAM_NAME = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

# `import light_bounces` gives the same names as the package (development scripts use it).
globals().update({_name: _value for _name, _value in vars(lightbounces).items()
                  if not _name.startswith("__")})

if __name__ == "__main__":
    main()
