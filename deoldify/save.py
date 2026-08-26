from fastai.learner import Learner, Callback
from fastai.vision.gan import GANLearner


class GANSaveCallback(Callback):
    """A `Callback` that saves history of metrics while training `learn` into CSV `filename`."""

    def __init__(
        self,
        learn: GANLearner,
        learn_gen: Learner,
        filename: str,
        save_iters: int = 1000,
    ):
        super().__init__()
        self.learn_gen = learn_gen
        self.filename = filename
        self.save_iters = save_iters

    def after_batch(self, **kwargs) -> None:
        iteration = kwargs.get('iteration', 0)
        epoch = kwargs.get('epoch', 0)
        if iteration == 0:
            return

        if iteration % self.save_iters == 0:
            self._save_gen_learner(iteration=iteration, epoch=epoch)

    def _save_gen_learner(self, iteration: int, epoch: int):
        filename = '{}_{}_{}'.format(self.filename, epoch, iteration)
        self.learn_gen.save(filename)
