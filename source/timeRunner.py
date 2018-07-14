#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import timeit
import click

N = 1000
d = 13
K = 8

number = 100

setup = """
import numpy as np
from likelihood import ScikitLL, SingleCoreLL

N = {N}
d = 13
K = 8

testX = np.random.random((N, d))
testMu = np.random.random((K, d))
testSigma = np.ones((K, d))
testWeights = np.ones(K) / K

eval = {LL}(testX, K)
"""

runs = "eval.loglikelihood(testMu, testSigma, testWeights)"

@click.command()
@click.option("--method",
              type=click.Choice(["ScikitLL", "SingleCoreLL"]),
              default="ScikitLL")
def main(method):
    print(f"{method} (100 iterations)")
    print("# pow, N, tot_time(s), scaled_time (us)")
    for Npow in range(2, 7):
        N = 10 ** Npow
        rtime = timeit.timeit(runs, setup.format(N=N, LL=method),
                              number=100)
        print(f"{Npow}, {N}, {rtime:.2f}, {rtime/N * 10**4:.2f}")
    print()


if __name__ == '__main__':
    main()
