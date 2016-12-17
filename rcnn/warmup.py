from mxnet.lr_scheduler import LRScheduler
import logging

class WarmupScheduler(LRScheduler):
    """Reduce learning rate in factor
    when step is less than warmup_step, then it will use warmup_lr,
    or, assume the weight has been updated by n times, then the learning
    rate will be

    base_lr * factor^(floor(n/step))

    Parameters
    ----------
    step: int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    warmup_lr: float
        the learning rate in warmup phase
    warmup_step: int
        the step of warm up phase
    """
    def __init__(self, step, factor=1, warmup_lr=1e-5, warmup_step=500):
        super(WarmupScheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.factor = factor
        self.count = 0
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        if num_update == 0:
            self.normal_lr = self.base_lr  # save base_lr, will used after warmup
        elif num_update < self.warmup_step:
            self.base_lr = self.warmup_lr
        elif num_update == self.warmup_step+1:
            self.warmup_step -= 1  # avoid Repeat logging
            self.base_lr = self.normal_lr
            logging.info("warmup is over: Change learning rate to %0.5e",
                         self.base_lr)
        elif num_update > self.warmup_step:
            if num_update - self.warmup_step > self.count + self.step:
                self.count += self.step
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
        return self.base_lr