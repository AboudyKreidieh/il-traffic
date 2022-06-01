"""Implementation of the InfoGAIL algorithm.

See: https://arxiv.org/pdf/1703.08840.pdf
"""
from il_traffic.algorithms.base import ILAlgorithm


class InfoGAIL(ILAlgorithm):
    """InfoGAIL training algorithm.

    See: https://arxiv.org/pdf/1703.08840.pdf
    """

    def _train(self):
        """Perform a number of optimization steps.

        Returns
        -------
        list of array_like
            the losses associated with the actions from each training step, and
            each ensemble
        """
        pass  # TODO
