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

        invCovs = 1 / diagCovs

        for i in range(self.numPoints):
            for mixes in range(numMixtures):
                tp = self.Xpoints[i] - means[mixes]
                temp = np.dot(tp ** 2, invCovs[mixes])
                temp *= -0.5
                ll[i] += weights[mixes] * np.exp(temp) * CovDet[mixes]

            ll[i] = np.log(ll[i]) - constMulti

        return np.sum(ll)


# ensure this remains the case for backwards compatibility
SingleCoreLL = SingleCoreLLSlow


class SingleCoreLLFast(SingleCoreLLSlow):
    def loglikelihood(self, means, diagCovs, weights):
        numMixtures = self.numMixtures
        assert (means.shape == (numMixtures, self.dim))
        assert (diagCovs.shape == (numMixtures, self.dim))
        assert (len(weights) == numMixtures)

        constMulti = self.dim / 2.0 * np.log(2 * np.pi)
        CovDet = np.zeros(numMixtures)
        for i in range(numMixtures):
            CovDet[i] = 1.0 / np.sqrt(np.prod(diagCovs[i]))

        llMat = np.zeros((self.numPoints, numMixtures))

        # XP = (N, d)
        # means & covars = (K, d)
        # weights = (K,)

        for mix in range(numMixtures):
            # this is the normal pdf exponent. Since Xp is (N,d) and means[mix] and diagCovs[mix]
            # are (d,) arrays, these operations are applied columnwise.
            tp = (self.Xpoints - means[mix]) / np.sqrt(diagCovs[mix])
            # tp is now an (N, d) matrix. I need to calculate the norm of this value for each n.
            # This is done by squaring and summing across axis 1 which sums the matrix across it's columns
            exponent = np.sum(tp ** 2, 1)
            # exponent is now an (N,) array.
            # we now calculate the weight * normalpdf which is where the below rhs does
            # this is a (N,) array too. We need to sum these up for each n across the mixtures
            # making a list of these values wouldn't work since we need to add the first item of the array
            # from each list, take the log and record the sum. This would force a N length loop which
            # we're trying to avoid.
            # Insread we generate an empty array (N, K) and populate it with the result as below
            llMat[:, mix] = weights[mix] * CovDet[mix] * np.exp(-0.5 * exponent)
        # the np.sum(llMat, 1) sums across columns as before, we then take the log of that value
        # and then sum it.
        # we have a constant offset generated from expanding the likelihood a little.
        # this is then subrtracted for consistency with other methods
        # overall this method is only 1 to 1.5x slower than the scikit fast version
        return np.sum(np.log(np.sum(llMat, 1))) - self.numPoints * constMulti
