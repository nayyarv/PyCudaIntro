#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import pytest
import numpy as np
from .cudaLL import chooseGridThread


@pytest.mark.parametrize("n,threads,blocks", (
        (100, 32, 4),
        (1000, 32, 32),
        (10000, 64, 157),
        (100000, 256, 391),
        (1000000, 512, 1954),

))
def test_chooser(n, threads, blocks):
    assert chooseGridThread(n) == (blocks, threads)


def test_GPUV1():
    """
    Simple evaluation of X = (1000 x 10) and K = 5.
    Seed the random generator

    Returns:

    """
    from .cudaLL import GPU_LL
    N = 1000
    d = 10
    K = 5

    np.random.seed(seed=103)

    xp = np.random.random((N, d))
    g = GPU_LL(xp, K)

    mu = np.random.random((K, d)).astype(np.float32)
    sig = np.ones((K, d)).astype(np.float32)
    weight = (np.ones(K) / K).astype(np.float32)

    ll = g.loglikelihood(mu, sig, weight)
    print(ll)

    ex = -9947.695
    assert abs(ll - ex) < 0.001

    # ensure randomness again
    np.random.seed()


