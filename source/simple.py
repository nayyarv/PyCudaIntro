# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np
from source import LikelihoodEvaluator


class SingleCoreLL(LikelihoodEvaluator):

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

        for i in range(self.numPoints):
            for mixes in range(numMixtures):
                temp = np.dot((self.Xpoints[i] - means[mixes]) / diagCovs[mixes],
                              (self.Xpoints[i] - means[mixes]))
                temp *= -0.5
                ll[i] += weights[mixes] * np.exp(temp) * CovDet[mixes]

            ll[i] = np.log(ll[i]) - constMulti

        return np.sum(ll)
