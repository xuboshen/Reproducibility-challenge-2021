import torch
import copy
import numpy as np
import random
import tarfile
import zipfile
from collections import defaultdict
import torchvision.transforms as T
from PIL import Image
import os.path as osp
import os
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import gdown
from utils.tools import read_image, check_isfile
from torchvision.transforms import(
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
    RandomApply
    #, GaussianBlur
    , RandomGrayscale, RandomResizedCrop,
    RandomHorizontalFlip
)
# sampler && dataloader && dataset && transform
AVAI_CHOICES = [
    'random_flip', 'random_resized_crop', 'normalize', 'instance_norm',
    'random_crop', 'random_translation', 'center_crop', 'cutout',
    'imagenet_policy', 'cifar10_policy', 'svhn_policy', 'randaugment',
    'randaugment_fixmatch', 'randaugment2', 'gaussian_noise', 'colorjitter',
    'randomgrayscale', 'gaussian_blur'
]

INTERPOLATION_MODES = {
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'nearest': Image.NEAREST
}

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=0, classname=''):
        assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = '' # the directory where the dataset is stored
    domains = [] # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x # labeled training data
        self._train_u = train_u # unlabeled training data (optional)
        self._val = val # validation data (optional)
        self._test = test # test data

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output

class RandomDomainSampler(Sampler):
    """Randomly samples N domains each with K images
    to form a minibatch of size N*K.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
        n_domain (int): number of domains to sample in a minibatch.
    """

    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())

        # Make sure each domain has equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class SeqDomainSampler(Sampler):
    """Sequential domain sampler, which randomly samples K
    images from each domain to form a minibatch.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        # Make sure each domain has equal number of images
        n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            for domain in self.domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomClassSampler(Sampler):
    """Randomly samples N classes each with K instances to
    form a minibatch of size N*K.

    Modified from https://github.com/KaiyangZhou/deep-person-reid.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
        n_ins (int): number of instances per class to sample in a minibatch.
    """

    def __init__(self, data_source, batch_size, n_ins):
        if batch_size < n_ins:
            raise ValueError(
                'batch_size={} must be no less '
                'than n_ins={}'.format(batch_size, n_ins)
            )
        self.data_source = data_source
        self.batch_size = batch_size
        self.n_ins = n_ins
        self.ncls_per_batch = self.batch_size // self.n_ins
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            self.index_dic[item.label].append(index)
        self.labels = list(self.index_dic.keys())
        print(len(self.labels), self.ncls_per_batch)
        assert len(self.labels) >= self.ncls_per_batch

        # estimate number of images in an epoch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.index_dic[label])
            if len(idxs) < self.n_ins:
                idxs = np.random.choice(idxs, size=self.n_ins, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.n_ins:
                    batch_idxs_dict[label].append(batch_idxs)
                    batch_idxs = []

        avai_labels = copy.deepcopy(self.labels)
        final_idxs = []

        while len(avai_labels) >= self.ncls_per_batch:
            selected_labels = random.sample(avai_labels, self.ncls_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)

        return iter(final_idxs)

    def __len__(self):
        return self.length

def build_sampler(
    sampler_type,
    cfg=None,
    data_source=None,
    batch_size=32,
    n_domain=0,
    n_ins=16
):
    if sampler_type == 'RandomSampler':
        return RandomSampler(data_source)

    elif sampler_type == 'SequentialSampler':
        return SequentialSampler(data_source)

    elif sampler_type == 'RandomDomainSampler':
        return RandomDomainSampler(data_source, batch_size, n_domain)

    elif sampler_type == 'SeqDomainSampler':
        return SeqDomainSampler(data_source, batch_size)

    elif sampler_type == 'RandomClassSampler':
        print(batch_size, n_ins)
        return RandomClassSampler(data_source, batch_size, n_ins)

    else:
        raise ValueError('Unknown sampler type: {}'.format(sampler_type))

def build_data_loader(
    cfg,
    sampler_type='SequentialSampler',
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=1,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader

class PACS(DatasetBase):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """
    dataset_dir = 'pacs'
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    data_url = 'https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE'
    # the following images contain errors and should be ignored
    _error_paths = ['sketch/dog/n02103406_4068-1.png']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.root))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, 'pacs.zip')
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.source_domains, cfg.target_domains
        )

        train = self._read_data(cfg.source_domains, 'train')
        val = self._read_data(cfg.source_domains, 'crossval')
        test = self._read_data(cfg.target_domains, 'all')

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            if split == 'all':
                file_train = osp.join(
                    self.split_dir, dname + '_train_kfold.txt'
                )
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(
                    self.split_dir, dname + '_crossval_kfold.txt'
                )
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(
                    self.split_dir, dname + '_' + split + '_kfold.txt'
                )
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                classname = impath.split('/')[-2]
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname
                )
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(' ')
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

        return items

class Random2DTranslation:
    """Given an image of (height, width), we resize it to
    (height*1.125, width*1.125), and then perform random cropping.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)

        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )

        return croped_img
def build_transform(cfg, is_train=True, choices=None):
    """Build transformation function.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
        choices (list, optional): list of strings which will overwrite
            cfg.INPUT.TRANSFORMS if given. Default is None.
    """
#     if cfg.INPUT.NO_TRANSFORM:
#         print('Note: no transform is applied!')
#         return None

    if choices is None:
        choices = ['random_flip', 'random_translation', 'normalize']

    for choice in choices:
        assert choice in AVAI_CHOICES, \
            'Invalid transform choice ({}), ' \
            'expected to be one of {}'.format(choice, AVAI_CHOICES)

    expected_size = '{}x{}'.format((224, 224)[0], (224, 224)[1])

    normalize = Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

    if is_train:
        return _build_transform_train(cfg, choices, expected_size, normalize)
    else:
        return _build_transform_test(cfg, choices, expected_size, normalize)

def _build_transform_train(cfg, choices, expected_size, normalize):
    print('Building transform_train')
    tfm_train = []

    interp_mode = INTERPOLATION_MODES['bilinear']

    print('+ resize to {}'.format(expected_size))
    tfm_train += [Resize((224, 224), interpolation=interp_mode)]

    if 'random_flip' in choices:
        print('+ random flip')
        tfm_train += [RandomHorizontalFlip()]

    if 'random_translation' in choices:
        print('+ random translation')
        tfm_train += [
            Random2DTranslation((224, 224)[0], (224, 224)[1])
        ]

    if 'random_crop' in choices:
        crop_padding = cfg.INPUT.CROP_PADDING
        print('+ random crop (padding = {})'.format(crop_padding))
        tfm_train += [RandomCrop((224, 224), padding=crop_padding)]

    if 'random_resized_crop' in choices:
        print('+ random resized crop')
        tfm_train += [
            RandomResizedCrop(cfg.INPUT.SIZE, interpolation=interp_mode)
        ]

    if 'center_crop' in choices:
        print('+ center crop (on 1.125x enlarged input)')
        enlarged_size = [int(x * 1.125) for x in (224, 224)]
        tfm_train += [Resize(enlarged_size, interpolation=interp_mode)]
        tfm_train += [CenterCrop((224, 224))]

    if 'imagenet_policy' in choices:
        print('+ imagenet policy')
        tfm_train += [ImageNetPolicy()]

    if 'cifar10_policy' in choices:
        print('+ cifar10 policy')
        tfm_train += [CIFAR10Policy()]

    if 'svhn_policy' in choices:
        print('+ svhn policy')
        tfm_train += [SVHNPolicy()]

    if 'randaugment' in choices:
        n_ = 2
        m_ = 10
        print('+ randaugment (n={}, m={})'.format(n_, m_))
        tfm_train += [RandAugment(n_, m_)]

    if 'randaugment_fixmatch' in choices:
        n_ = 2
        print('+ randaugment_fixmatch (n={})'.format(n_))
        tfm_train += [RandAugmentFixMatch(n_)]

    if 'randaugment2' in choices:
        n_ = 2
        print('+ randaugment2 (n={})'.format(n_))
        tfm_train += [RandAugment2(n_)]

    if 'colorjitter' in choices:
        print('+ color jitter')
        tfm_train += [
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )
        ]

    if 'randomgrayscale' in choices:
        print('+ random gray scale')
        tfm_train += [RandomGrayscale(p=cfg.INPUT.RGS_P)]

    if 'gaussian_blur' in choices:
        # print(f'+ gaussian blur (kernel={cfg.INPUT.GB_K})')
        # tfm_train += [
        #     RandomApply([GaussianBlur(cfg.INPUT.GB_K)], p=cfg.INPUT.GB_P)
        # ]
        print("there's no gaussian blur in torchvision == 1.5.0, sorry about that")

    print('+ to torch tensor of range [0, 1]')
    tfm_train += [ToTensor()]

    if 'normalize' in choices:
        print(
            '+ normalization (mean={}, '
            'std={})'.format([0., 0., 0.], [1., 1., 1.])
        )
        tfm_train += [normalize]

    if 'gaussian_noise' in choices:
        print(
            '+ gaussian noise (mean={}, std={})'.format(
                cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD
            )
        )
        tfm_train += [GaussianNoise(cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD)]

    if 'instance_norm' in choices:
        print('+ instance normalization')
        tfm_train += [InstanceNormalization()]

    tfm_train = Compose(tfm_train)

    return tfm_train

def _build_transform_test(cfg, choices, expected_size, normalize):
    print('Building transform_test')
    tfm_test = []

    interp_mode = INTERPOLATION_MODES['bilinear']

    print('+ resize to {}'.format(expected_size))
    tfm_test += [Resize((224, 224), interpolation=interp_mode)]

    if 'center_crop' in choices:
        print('+ center crop (on 1.125x enlarged input)')
        enlarged_size = [int(x * 1.125) for x in (224, 224)]
        tfm_test += [Resize(enlarged_size, interpolation=interp_mode)]
        tfm_test += [CenterCrop((224, 224))]

    print('+ to torch tensor of range [0, 1]')
    tfm_test += [ToTensor()]

    if 'normalize' in choices:
        print(
            '+ normalization (mean={}, '
            'std={})'.format([0., 0., 0.], [1., 1., 1.])
        )
        tfm_test += [normalize]

    if 'instance_norm' in choices:
        print('+ instance normalization')
        tfm_test += [InstanceNormalization()]

    tfm_test = Compose(tfm_test)

    return tfm_test

class DataManager:
    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = PACS(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print('* Using custom transform for training')
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print('* Using custom transform for testing')
            tfm_test = custom_tfm_test

        # Build train_loader_x
        print(tfm_train)
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.trainsampler,
            data_source=dataset.train_x,
            batch_size=cfg.batch_size,
            n_domain=cfg.n_domains,
            n_ins=cfg.n_ins,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # # Build val_loader
        # val_loader = None
        # if dataset.val:
        #     val_loader = build_data_loader(
        #         cfg,
        #         sampler_type=cfg.testsampler,
        #         data_source=dataset.val,
        #         batch_size=cfg.batch_size,
        #         tfm=tfm_test,
        #         is_train=False,
        #         dataset_wrapper=dataset_wrapper
        #     )

        # # Build test_loader
        # test_loader = build_data_loader(
        #     cfg,
        #     sampler_type=cfg.testsampler,
        #     data_source=dataset.test,
        #     batch_size=cfg.batch_size,
        #     tfm=tfm_test,
        #     is_train=False,
        #     dataset_wrapper=dataset_wrapper
        # )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.source_domains)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        # self.val_loader = val_loader
        # self.test_loader = test_loader

        if cfg.verbose:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        print('***** Dataset statistics *****')

        print('  Dataset: {}'.format(cfg.dataset))

        if cfg.source_domains:
            print('  Source domains: {}'.format(cfg.source_domains))
        if cfg.target_domains:
            print('  Target domains: {}'.format(cfg.target_domains))

        print('  # classes: {:,}'.format(self.num_classes))

        print('  # train_x: {:,}'.format(len(self.dataset.train_x)))

        if self.dataset.val:
            print('  # val: {:,}'.format(len(self.dataset.val)))

        print('  # test: {:,}'.format(len(self.dataset.test)))

class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = 1 if is_train else 1
        self.return_img0 = False

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES['bilinear']
        to_tensor = []
        to_tensor += [T.Resize((224, 224), interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if 'normalize' in ['random_flip', 'random_translation', 'normalize']:
            normalize = T.Normalize(
                mean=[0., 0., 0.], std=[1., 1., 1.]
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
