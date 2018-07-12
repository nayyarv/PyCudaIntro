#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import timeit
import numpy as np

from source.scikitLL import ScikitLL
from source.simple import SingleCoreLL

N = 100
d = 13
K = 8

number = 1000

setup = """
import numpy as np
from source.scikitLL import ScikitLL
from source.simple import SingleCoreLL

N = {N}
d = {d}
K = {K}

testX = np.random.random((N, d))
testMu = np.random.random((K, d))
testSigma = np.ones((K, d))
testWeights = np.ones(K) / K

eval = {LL}(testX, K)
"""

runs = "[eval.loglikelihood(testMu, testSigma, testWeights)]"


def main():
    for LL in ["ScikitLL", "SingleCoreLL"]:
        print(LL)
        print(timeit.timeit(
            runs, setup.format(LL=LL, N=N, d=d, K=K), number=number))


if __name__ == '__main__':
    main()
