# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import pytest
import numpy as np

# used in both, so we might as well import at top level
from likelihood import ScikitLL

N = 100
d = 13
K = 8

numTests = 4


def randMat(dim, N=numTests):
    return [np.random.random(dim) for _ in range(N)]


@pytest.fixture(scope="module", params=randMat((N, d)))
def likelihoods(request):
    from likelihood.simple import SingleCoreLL
    return [SingleCoreLL(request.param, K),
            ScikitLL(request.param, K)]


@pytest.fixture(scope="module", params=randMat((N, d)))
def fastlikelihoods(request):
    from likelihood.simple import SingleCoreLLFast
    return [SingleCoreLLFast(request.param, K),
            ScikitLL(request.param, K)]

@pytest.fixture(scope="module", params=randMat((N, d)))
def gpu_likelihoods(request):
    GPU_LL = pytest.importorskip('likelihood.GPULL')
    return [ScikitLL(request.param, K),
            GPU_LL(request.param, K)]


@pytest.mark.parametrize('means', randMat((K, d)))
@pytest.mark.parametrize('covars', randMat((K, d)))
@pytest.mark.parametrize('weights', randMat(K))
def test_SK_consistent(likelihoods, means, covars, weights):
    weights /= np.sum(weights)

    baseEval, scikitEval = likelihoods

    baseLL = baseEval.loglikelihood(means, covars, weights)
    scikitLL = scikitEval.loglikelihood(means, covars, weights)
    assert abs(baseLL - scikitLL) < 0.0001

@pytest.mark.parametrize('means', randMat((K, d)))
@pytest.mark.parametrize('covars', randMat((K, d)))
@pytest.mark.parametrize('weights', randMat(K))
def test_BaseFast_consistent(fastlikelihoods, means, covars, weights):
    weights /= np.sum(weights)

    baseFastEval, scikitEval = fastlikelihoods

    baseFastLL = baseFastEval.loglikelihood(means, covars, weights)
    scikitLL = scikitEval.loglikelihood(means, covars, weights)
    assert abs(baseFastLL - scikitLL) < 0.0001


@pytest.mark.parametrize('means', randMat((K, d)))
@pytest.mark.parametrize('covars', randMat((K, d)))
@pytest.mark.parametrize('weights', randMat(K))
def test_GPU_consistent(gpu_likelihoods, means, covars, weights):
    weights /= np.sum(weights)

    scikitEval, gpuEval = gpu_likelihoods

    scikitLL = scikitEval.loglikelihood(means, covars, weights)
    gpuLL = gpuEval.loglikelihood(means, covars, weights)
    # drop accuracy for float32
    assert abs(gpuLL - scikitLL) < 0.01
