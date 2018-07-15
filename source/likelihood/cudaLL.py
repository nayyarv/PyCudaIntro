#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np

import pycuda  # get import errors out of the way
from likelihood.base import LikelihoodEvaluator


def chooseGridThread(n):
    """
    Modify this function to change how we choose number of
    threads and blocks

    Args:
        n (int): number of datapoints

    Returns:
        (int, int): (nblocks, nthreads)
    """
    from math import ceil, sqrt
    nroot = sqrt(n)
    curr = 32
    while curr * 2 < nroot and curr < 2048:
        curr *= 2
    nthreads = curr

    nblocks = ceil(n / nthreads)
    return nblocks, nthreads


class GPU_LL(LikelihoodEvaluator):

    def __str__(self):
        return "GPU Implementation of GMM"

    @property
    def cuFile(self):
        import os
        return os.path.join(os.path.dirname(__file__), "kernel.cu")

    def __init__(self, Xpoints, numMixtures):
        super().__init__(Xpoints, numMixtures)

        import pycuda.autoinit  # must be run

        from pycuda import gpuarray
        from pycuda.compiler import SourceModule

        self.gpuarray = gpuarray
        self.numThreads, self.numBlocks = chooseGridThread(self.numPoints)
        with open(self.cuFile) as f:
            # have to use a replace here since we can't guarantee the format working
            # with the c syntax. A bit hacky sadly, but easier than mallocing
            mod = SourceModule(f.read().replace(
                "MAX_THREADS", str(self.numThreads)))

        print("numBlocks: {}, numPoints: {}".format(self.numBlocks, self.numPoints))
        # Set the right number of threads and blocks given the datasize

        self.likelihoodKernel = mod.get_function("likelihoodKernel")
        self.likelihoodKernel.prepare('PPPPiiiP')

        self.Xpoints = self.Xpoints.astype(np.float32)
        # dump the X in the GPU memory so we don't need to keep transferring
        # note we assume size(Xpoints) < GPU memory
        self.Xpoints = gpuarray.to_gpu_async(self.Xpoints)

        # we allocate memory for the parameters so this is only done on startup
        self.means_gpu = gpuarray.zeros(shape=(self.numMixtures, self.dim), dtype=np.float32)
        self.diagCovs_gpu = gpuarray.zeros(shape=(self.numMixtures, self.dim), dtype=np.float32)
        self.weights_gpu = gpuarray.zeros(shape=self.numMixtures, dtype=np.float32)

        # Allocate Memory for all our computations
        self.llVal = gpuarray.zeros(shape=self.numBlocks, dtype=np.float32)

    def loglikelihood(self, means, diagCovs, weights):
        # quick sanity checks
        assert (means.shape == (self.numMixtures, self.dim))
        assert (diagCovs.shape == (self.numMixtures, self.dim))
        assert (len(weights) == self.numMixtures)

        # convert to sp if needed
        if means.dtype != np.float32:
            means = means.astype(np.float32)
        if diagCovs.dtype != np.float32:
            diagCovs = diagCovs.astype(np.float32)
        if weights.dtype != np.float32:
            weights = weights.astype(np.float32)

        self.means_gpu.set_async(means)
        self.diagCovs_gpu.set_async(diagCovs)
        self.weights_gpu.set(weights)

        self.likelihoodKernel.prepared_call((self.numBlocks, 1), (self.numThreads, 1, 1),
                                            self.Xpoints.gpudata, self.means_gpu.gpudata,
                                            self.diagCovs_gpu.gpudata,
                                            self.weights_gpu.gpudata,
                                            self.dim, self.numPoints, self.numMixtures,
                                            self.llVal.gpudata)

        ll = self.gpuarray.sum(self.llVal).get()
        return ll