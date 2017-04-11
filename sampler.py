"""Tools for generating sequences from a probabilistic model"""
import itertools
import numpy as np
import random


class SequenceSampler(object):
    """Object for generating sequences from a probabilistic model"""
    def gen_sequence(self):
        """Generate sequences"""
        self.reset()
        sequence = list(
            itertools.takewhile(lambda x: x != self.end_token, self.gen_sample())
        )
        return sequence

    def gen_sample(self):
        """Generate a sample"""
        raise NotImplementedError

    def reset(self):
        """Reset internal state"""
        pass


class MCSampler(SequenceSampler):
    """Generate sequences from a first-order Markov model

    Note: n-th state always considered to be an end state.
    """
    def __init__(self, alpha, gamma):
        """Initialize model with predefined parameters

        args:
            alpha: np.array. Transition probabilities.
            gamma: np.array. Initial value probabilities.
        """
        assert alpha.ndim == 2, "ERROR: alpha not a matrix"
        assert alpha.shape[0] == alpha.shape[1], "ERROR: alpha not square"
        assert alpha.shape[0] == gamma.shape[0], "ERROR: incompatible alpha and gamma"

        self.alpha = alpha
        self.gamma = gamma
        self.end_token = alpha.shape[0] - 1
        self.prev = None

    @classmethod
    def random_init(cls, n, zero_diag=True):
        """Randomly initialize model parameters

        args:
            n: int. Number of states.
            zero_diag: bool. True if diagonal elements of transition matrix are
                forced to be zero (e.g. states do not repeat)
        """
        gamma = np.random.rand(n)
        gamma[n-1] = 0
        gamma = gamma / np.sum(gamma)

        alpha = np.random.rand(n, n)
        if zero_diag:
            np.fill_diagonal(alpha, 0)
        alpha = alpha / np.sum(alpha, axis=1).reshape((n, 1))

        return cls(alpha, gamma)

    def gen_sample(self):
        while True:
            if self.prev is None:
                prob = self.gamma
            else:
                prob = self.alpha[self.prev]
            cdf = np.cumsum(prob)
            rng = random.random()
            sample = np.argmax(cdf > rng)
            self.prev = sample
            yield sample

    def reset(self):
        self.state = None

