#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"


from .simple import SingleCoreLL, SingleCoreLLFast

try:
    from .scikitLL import ScikitLL
except ImportError:
    pass

try:
    from .cudaLL import GPU_LL as GPULL
except ImportError:
    pass

