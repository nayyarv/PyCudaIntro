# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import pytest
import numpy as np

from source.scikitLL import ScikitLL
from source.simple import SingleCoreLL

N = 100
d = 13
K = 8

numTests = 6


def randMat(dim, N=numTests):
    return [np.random.random(dim) for _ in range(N)]


@pytest.fixture(scope="module", params=randMat((N, d)))
def likelihoods(request):
    return [SingleCoreLL(request.param, K),
            ScikitLL(request.param, K)]


@pytest.mark.parametrize('means', randMat((K, d)))
@pytest.mark.parametrize('covars', randMat((K, d)))
@pytest.mark.parametrize('weights', randMat(K))
def test_consistent(likelihoods, means, covars, weights):
    weights /= np.sum(weights)

    baseEval, scikitEval = likelihoods

    baseLL = baseEval.loglikelihood(means, covars, weights)
    scikitLL = scikitEval.loglikelihood(means, covars, weights)
    assert abs(baseLL - scikitLL) < 0.0001
