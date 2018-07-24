# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np
from .base import LikelihoodEvaluator


class SingleCoreLLSlow(LikelihoodEvaluator):

    def __init__(self, Xpoints, numMixtures):
        super().__init__(Xpoints, numMixtures)

    def __str__(self):
        return "Single Core Implementation of GMM"

    __repr__ = __str__

    def loglikelihood(self, means, diagCovs, weights):
        numMixtures = self.numMixtures

        # update if need be

        assert (means.shape == (numMixtures, self.dim))
        assert (diagCovs.shape == (numMixtures, self.dim))
        assert (len(weights) == numMixtures)

        numMixtures = len(weights)

        ll = np.zeros(self.numPoints)

        constMulti = self.dim / 2.0 * np.log(2 * np.pi)

        CovDet = np.zeros(numMixtures)

        for i in range(numMixtures):
            CovDet[i] = 1.0 / np.sqrt(np.prod(diagCovs[i]))

        invCovs = 1/diagCovs

        for i in range(self.numPoints):
            for mixes in range(numMixtures):
                tp = self.Xpoints[i] - means[mixes]
                temp =  np.dot(tp**2, invCovs[mixes])
                temp *= -0.5
                ll[i] += weights[mixes] * np.exp(temp) * CovDet[mixes]

            ll[i] = np.log(ll[i]) - constMulti

        return np.sum(ll)











class SingleCoreLLFast(SingleCoreLLSlow):
    def loglikelihood(self, means, diagCovs, weights):
        numMixtures = self.numMixtures
        assert (means.shape == (numMixtures, self.dim))
        assert (diagCovs.shape == (numMixtures, self.dim))
        assert (len(weights) == numMixtures)

        ll = np.zeros(self.numPoints)
        constMulti = self.dim / 2.0 * np.log(2 * np.pi)
        CovDet = np.zeros(numMixtures)
        for i in range(numMixtures):
            CovDet[i] = 1.0 / np.sqrt(np.prod(diagCovs[i]))
      
        llMat = np.zeros((self.numPoints, numMixtures))

        for mix in range(numMixtures):
            # this is the normal pdf exponent
            tp = (self.Xpoints - means[mix]) / np.sqrt(diagCovs[mix])
            # this is the dot products of every row of tp dotted with itself. 
            # It's a row vector of length numPoints.
            # np.dot is matmul for 2d matrices so np.diag(np.dot(tp, tp.T)) does n^2
            # unecessary computations. The einsum code is the real secret sauce of this 
            # code here.
            exponent = np.einsum('ij,ij->i', tp , tp)
            # exponent = np.diag(np.dot(tp, tp.T))
            llMat[:, mix] = weights[mix] * CovDet[mix] * np.exp(-0.5 * exponent)
        # sum all rows
        return np.sum(np.log(np.sum(llMat, 1))) - self.numPoints * constMulti



SingleCoreLL = SingleCoreLLSlow
