"""Implementation of the DAgger algorithm.

See: https://arxiv.org/pdf/1011.0686.pdf
"""
import numpy as np
from tqdm import tqdm

from il_traffic.algorithms.base import ILAlgorithm


class DAgger(ILAlgorithm):
    """DAgger training algorithm.

    See: https://arxiv.org/pdf/1011.0686.pdf
    """

    def _train(self):
        """Perform a number of optimization steps.

        Returns
        -------
        list of array_like
            the losses associated with the actions from each training step, and
            each ensemble
        """
        # Covert samples to numpy array.
        observations = np.array(self.samples["obs"])
        actions = np.array(self.samples["actions"])

        loss = []
        for _ in tqdm(range(self.num_train_steps)):
            # Sample a batch.
            batch_i = [np.random.randint(0, len(actions), size=self.batch_size)
                       for _ in range(self.model.num_ensembles)]

            # Run the training step.
            loss_step = self.model.train(
                obs=[observations[batch_i[i], :]
                     for i in range(self.model.num_ensembles)],
                action=[actions[batch_i[i], :]
                        for i in range(self.model.num_ensembles)],
            )
            loss.append(loss_step)

        return loss
