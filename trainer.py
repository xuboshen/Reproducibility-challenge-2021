import time
import numpy as np
import os.path as osp
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import functools

from collections import OrderedDict
from data import DataManager

from utils import *
from optimizer import *
from evaluator import *
from models import *
from utils.utils import compute_accuracy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.switch_backend('Agg')
#  && SimpleNet && trainerbase && evaluator
## trainerbase construction

def visualize(data, label, figname='embedding'):
    print('plotting')
    fig = plt.figure()
    ax = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    # type5_x = []
    # type5_y = []
    # type6_x = []
    # type6_y = []
    # type7_x = []
    # type7_y = []
    for i in range(len(data)):
        if label[i] == 0:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if label[i] == 1:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
        if label[i] == 2:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])
        if label[i] == 3:
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])
        # if label[i] == 4:
        #     type4_x.append(data[i][0])
        #     type4_y.append(data[i][1])
        # if label[i] == 5:
        #     type5_x.append(data[i][0])
        #     type5_y.append(data[i][1])
        # if label[i] == 6:
        #     type6_x.append(data[i][0])
        #     type6_y.append(data[i][1])

    type1 = plt.scatter(type1_x, type1_y, s=1, c='r')
    type2 = plt.scatter(type2_x, type2_y, s=1, c='g')
    type3 = plt.scatter(type3_x, type3_y, s=1, c='b')
    type4 = plt.scatter(type4_x, type4_y, s=1, c='yellow')
    # type5 = plt.scatter(type5_x, type5_y, s=1, c='pink')
    # type6 = plt.scatter(type6_x, type6_y, s=1, c='black')
    # type7 = plt.scatter(type7_x, type7_y, s=1, c='purple')
    plt.legend((type1, type2, type3, type4
    # , type5, type6, type7
    ), ('Art', 'Cartoon', 'Photo', 'Sketch'),
              #  ('dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'),
               loc=(0.97, 0.5))
    plt.xticks()
    plt.yticks()
    print('plot done')
    plt.show()
    plt.savefig('{}.png'.format(figname), dpi=200, bbox_inches='tight')
    return fig

class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        
    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        assert name not in self._models, 'Found duplicate model names'

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched
    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, directory, is_best=False, model_name=''):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    'state_dict': model_dict,
                    'epoch': epoch + 1,
                    'optimizer': optim_dict,
                    'scheduler': sched_dict
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print('No checkpoint found, train from scratch')
            return 0

        print(
            'Found checkpoint in "{}". Will resume training'.format(directory)
        )

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode='train', names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Loss is infinite or NaN!')

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch()
            # self.after_epoch()
        self.after_train()

    def before_train(self):
        directory = self.cfg.output_dir
        if self.cfg.resume:
            directory = self.cfg.resume
        self.start_epoch = self.resume_model_if_exist(directory)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        meet_checkpoint_freq = (
            self.epoch + 1
        ) % self.cfg.checkpoint_freq == 0 if self.cfg.checkpoint_freq > 0 else False
        self.save_model(
            self.epoch,
            self.output_dir,
            model_name='model-best.pth.tar'
        )
        print('Finished training')
        self.test('all')
        do_test = True
        if do_test:
            print('before loading the best val model')
            self.test('crossval')
            if self.cfg.final_model == 'best_val':
                print('Deploy the model with the best val performance')
                self.load_model(self.output_dir)
            self.test('all')

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.no_test
        meet_checkpoint_freq = (
            self.epoch + 1
        ) % self.cfg.checkpoint_freq == 0 if self.cfg.checkpoint_freq > 0 else False

        if do_test and self.cfg.final_model == 'best_val':
            curr_result = self.test(split='crossval')
            # self.test(split='all')
            is_best = curr_result > self.best_result
            print('curr_result')
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name='model-best.pth.tar'
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.print_freq == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx

            end = time.time()

    @torch.no_grad()
    def test(self, split='train'):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        if split is None:
            split = self.cfg.train_split
        if split == 'train':
          data_loader = self.train_loader_x
        # if split == 'crossval' and self.val_loader is not None:
        #     data_loader = self.val_loader
        #     print('Do evaluation on {} set'.format(split))
        # else:
        #     data_loader = self.test_loader
        #     print('Do evaluation on test set')
        total_feature = []
        total_domain = []
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= 40:
              break
            input, label, domain = self.parse_batch_train(batch)
            output, feature = self.model_inference(input)
            if len(total_domain) == 0:
              total_feature = feature.cpu().numpy()
              total_domain = domain.cpu().numpy()
            else:
              total_feature = np.concatenate((total_feature, feature.cpu().numpy()), axis=0)
              total_domain = np.concatenate((total_domain, domain.cpu().numpy()), axis=0)
            # self.evaluator.process(output, label)
        print(total_feature.shape)
        total_feature = total_feature.reshape(1280*4, -1)
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(total_feature.reshape((1280*4, -1)))
        
        visualize(result, total_domain.reshape(-1), 'statistics_domain_origin')
        # results = self.evaluator.evaluate()
        return None
        # return list(results.values())[0]

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        domain = batch['domain']

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

    def parse_batch_test(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        return self.F(input, True)

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

class Mytrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        if torch.cuda.is_available() and cfg.gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Save as attributes some frequently used variables
        self.start_epoch = 0
        self.max_epoch = cfg.max_epoch
        self.output_dir = cfg.output_dir
        self.cfg = cfg
        self.best_result = -np.inf
        self.build_data_loader()
        self.build_model()
        self.evaluator = Classification(cfg, lab2cname=self.dm.lab2cname)
        self.batch_size = cfg.batch_size

    def build_data_loader(self):
        """Create essential data-related attributes.

        What must be done in the re-implementation
        of this method:
        1) initialize data manager
        2) assign as attributes the data loaders
        3) assign as attribute the number of classes
        """
        self.dm = DataManager(self.cfg)
        self.train_loader_x = self.dm.train_loader_x
        # self.val_loader = self.dm.val_loader
        # self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
    
    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        print('Building model')
        self.F = SimpleNet(self.cfg, self.cfg, self.num_classes).to(self.device)
        
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, self.cfg)
        self.sched_F = build_lr_scheduler(self.optim_F, self.cfg)
        self.register_model('F', self.F, self.optim_F, self.sched_F)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Detected {device_count} GPUs. Activate multi-gpu training')
            self.G = nn.DataParallel(self.G)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]['lr']

    def forward_backward(self, batch):
        loss = 0.0
        input, label, domain = self.parse_batch_train(batch)
        out_cls = self.F(input)
        loss += F.cross_entropy(out_cls, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss':loss.item(),
            'acc': compute_accuracy(self.F(input), label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary



class Drawtrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        if torch.cuda.is_available() and cfg.gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Save as attributes some frequently used variables
        self.start_epoch = 0
        self.max_epoch = cfg.max_epoch
        self.output_dir = cfg.output_dir
        self.cfg = cfg
        self.best_result = -np.inf
        self.build_data_loader()
        self.build_model()
        self.evaluator = Classification(cfg, lab2cname=self.dm.lab2cname)
        self.batch_size = cfg.batch_size

    def build_data_loader(self):
        """Create essential data-related attributes.

        What must be done in the re-implementation
        of this method:
        1) initialize data manager
        2) assign as attributes the data loaders
        3) assign as attribute the number of classes
        """
        self.dm = DataManager(self.cfg)
        self.train_loader_x = self.dm.train_loader_x
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
    
    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        print('Building model')
        self.F = SimpleNet(self.cfg, self.cfg, self.num_classes).to(self.device)
        
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, self.cfg)
        self.sched_F = build_lr_scheduler(self.optim_F, self.cfg)
        self.register_model('F', self.F, self.optim_F, self.sched_F)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Detected {device_count} GPUs. Activate multi-gpu training')
            self.G = nn.DataParallel(self.G)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]['lr']

    def forward_backward(self, batch):
        loss = 0.0
        input, label, domain = self.parse_batch_train(batch)
        out_cls = self.F(input)
        loss += F.cross_entropy(out_cls, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss':loss.item(),
            'acc': compute_accuracy(self.F(input), label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary























