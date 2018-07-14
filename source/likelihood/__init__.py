#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"


from .simple import SingleCoreLL

try:
    from .scikitLL import ScikitLL
except ImportError:
    pass
