# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

from .base import LikelihoodEvaluator

import numpy as np
# import at top level so we have import errors earlier than later
from sklearn.mixture.gaussian_mixture import GaussianMixture, _compute_precision_cholesky


class ScikitLL(LikelihoodEvaluator):
    """
    Fastest Single Core Version so far!
    """

    def __init__(self, Xpoints, numMixtures):
        super().__init__(Xpoints, numMixtures)
        self.evaluator = GaussianMixture(numMixtures, 'diag')
        self.Xpoints = Xpoints
        self.evaluator.fit(Xpoints)

    def __str__(self):
        return "SciKit's learn implementation Implementation"

    def loglikelihood(self, means, diagCovs, weights):
        self.evaluator.weights_ = weights
        self.evaluator.covariances_ = diagCovs
        self.evaluator.means_ = means
        self.evaluator.precisions_cholesky_ = _compute_precision_cholesky(diagCovs, "diag")

        return self.numPoints * np.sum(self.evaluator.score(self.Xpoints))
