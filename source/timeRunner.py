#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"


import timeit
import numpy as np

from source.scikitLL import ScikitLL
from source.simple import SingleCoreLL


setup = """
import numpy as np
from source.scikitLL import ScikitLL
from source.simple import SingleCoreLL

N = 100
d = 13
K = 8

testX = np.random.random((N, d))
testMu = np.random.random((K, d))
testSigma = np.ones((K, d))
testWeights = np.ones(K) / K

eval = {}(testX, K)
"""

runs = "[eval.loglikelihood(testMu, testSigma, testWeights)]"


def main():
    for LL in ["ScikitLL", "SingleCoreLL"]:
        print(LL)
        print(timeit.timeit(runs, setup.format(LL), number=1000))


if __name__ == '__main__':
    main()

