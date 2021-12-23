import torch
from collections import OrderedDict, defaultdict

## evaluatorbase
class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

## classification for evaluation
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def reset(self):
        self._correct = 0
        self._total = 0

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

    def evaluate(self):
        results = OrderedDict()
        acc = 100. * self._correct / self._total
        err = 100. - acc
        # The first value will be returned by trainer.test()
        results['accuracy'] = acc
        results['error_rate'] = err

        print(
            '=> result\n'
            '* total: {:,}\n'
            '* correct: {:,}\n'
            '* accuracy: {:.2f}%\n'
            '* error: {:.2f}%'.format(self._total, self._correct, acc, err)
        )
        return results
