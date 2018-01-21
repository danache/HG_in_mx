import mxnet as mx
import numpy as np
class MapMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MapMetric, self).__init__('MultiBox')
        self.eps = eps
        self.num = 1
        self.name = ['MSE']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0]
            self.sum_metric = [0.0]

    def update(self, labels, preds):

        lab = labels[0].asnumpy()
        pr = preds[0].asnumpy()

        self.sum_metric[0] += np.sum(np.power(lab-pr,2)) / lab.shape[0]
        self.num_inst[0] += lab.shape[0]


    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)
