#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np

import pycuda  # get import errors out of the way
from source import LikelihoodEvaluator


class GPULL(LikelihoodEvaluator):

    def __str__(self):
        return "GPU Implementation of GMM"

    def __init__(self, Xpoints, numMixtures):
        super().__init__(Xpoints, numMixtures)

        import pycuda.autoinit
        from pycuda import gpuarray
        from pycuda.compiler import SourceModule

        self.gpuarray = gpuarray

        with open("kernel.cu") as f:

            if self.numPoints >= 1024:
                mod = SourceModule(f.read().format(MAX_THREADS=1024))
                self.numThreads = 1024
            else:
                mod = SourceModule(f.read().format(MAX_THREADS=self.numPoints))
                self.numThreads = self.numPoints

        if self.numPoints > self.numThreads:
            self.numBlocks = self.numPoints / self.numThreads
            if self.numPoints % self.numThreads != 0: self.numBlocks += 1
        else:
            self.numBlocks = 1

        print("numBlocks: {}, numPoints: {}".format(self.numBlocks, self.numPoints))
        # Set the right number of threads and blocks given the datasize
        # Using a max of 1024 threads, fix correct blocksize

        self.likelihoodKernel = mod.get_function("likelihoodKernel")
        self.likelihoodKernel.prepare('PPPPiiiP')

        self.Xpoints = self.Xpoints.astype(np.float32)
        self.Xpoints = gpuarray.to_gpu_async(self.Xpoints)

        self.means_gpu = gpuarray.zeros(shape=(self.numMixtures, self.dim), dtype=np.float32)
        self.diagCovs_gpu = gpuarray.zeros(shape=(self.numMixtures, self.dim), dtype=np.float32)
        self.weights_gpu = gpuarray.zeros(shape=self.numMixtures, dtype=np.float32)

        self.llVal = gpuarray.zeros(shape=self.numBlocks, dtype=np.float32)

        # Allocate Memory for all our computations

    def loglikelihood(self, means, diagCovs, weights):

        assert (means.shape == (self.numMixtures, self.dim))
        assert (diagCovs.shape == (self.numMixtures, self.dim))
        assert (len(weights) == self.numMixtures)

        if means.dtype != np.float32:
            means = means.astype(np.float32)
        if diagCovs.dtype != np.float32:
            diagCovs = diagCovs.astype(np.float32)
        if weights.dtype != np.float32:
            weights = weights.astype(np.float32)

        # quick sanity checks
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
