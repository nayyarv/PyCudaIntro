# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

"""
inherit from this/duck type this object 

N = number of points
d = dimensions
K = number of mixtures
"""


class LikelihoodEvaluator:

    def __init__(self, Xpoints, numMixtures):
        """

        Args:
            Xpoints (np.array (N,d)): dataset
            numMixtures (int): number of mixtures to train on
        """
        assert (Xpoints.ndim == 2)
        self.Xpoints = Xpoints
        self.numPoints, self.dim = Xpoints.shape
        self.numMixtures = numMixtures

    def loglikelihood(self, means, diagCovs, weights):
        """

        Args:
            means (np.array (K,d)):
            diagCovs (np.array (K,d)):
            weights (np.array (K)): weights must sum up to 1

        Returns:

        """
        raise NotImplementedError

    __call__ = loglikelihood


try:
    from sklearn.mixture import GaussianMixture
    import source.scikitLL

    Likelihood = source.scikitLL.ScikitLL
except ImportError:
    import source.simple

    Likelihood = source.simple.SingleCoreLL
