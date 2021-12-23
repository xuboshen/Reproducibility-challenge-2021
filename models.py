import torch
import torch.nn as nn
from backbone import *
from backbone import resnet18_ms_l2
import matplotlib.pyplot as plt
class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = resnet18(True)
        fdim = self.backbone.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            y, f = self.backbone(x, True)
            return y, f

        return y

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, conv_dim = 64, c_dim = 5, repeat_num = 6):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Projector(nn.Module):
    def __init__(
        self,
        in_features=2048,
        hidden_layers=[],
        activation='relu',
        bn=True,
        dropout=0.,
        num_classes=None
    ):
        super().__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        assert len(hidden_layers) > 0
        self.out_features = hidden_layers[-1]
        self.num_classes=num_classes
        mlp = []

        if activation == 'relu':
            act_fn = functools.partial(nn.ReLU, inplace=True)
        elif activation == 'leaky_relu':
            act_fn = functools.partial(nn.LeakyReLU, inplace=True)
        else:
            raise NotImplementedError

        for hidden_dim in hidden_layers:
            mlp += [nn.Linear(in_features, hidden_dim)]
            if bn:
                mlp += [nn.BatchNorm1d(hidden_dim)]
            mlp += [act_fn()]
            if dropout > 0:
                mlp += [nn.Dropout(dropout)]
            in_features = hidden_dim

        self.projector = nn.Sequential(*mlp)
        if num_classes is not None:
            self.classifier = nn.Linear(hidden_layers[-1], num_classes)

    def forward(self, x, constant=0, reverse=False):
        output = self.projector(x)
        if reverse is True:
            x = GradReverse.grad_reverse(x, constant)
        if self.num_classes is not None:
            output = self.classifier(output)
        return output
    


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class DomianClassifier(nn.Module):
    def __init__(self, conv_dim=64, domain=3, repeat_num=6):
        super(DomianClassifier, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            # layers.append(nn.BatchNorm2d(curr_dim * 2, momentum=0.9, eps=1e-5))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        self.fc = nn.Conv2d(curr_dim, domain, kernel_size=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        h = self.pool(h)
        # out_src = self.conv1(h)
        out_cls = self.fc(h)
        out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
        # out_cls = F.softmax(out_cls,dim=1)
        return out_cls

class classifier(nn.Module):
    def __init__(self, in_features=512, num_classes=None):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
    def forward(self, x, constant=0, reverse=False):
        if reverse is True:
            x = GradReverse.grad_reverse(x, constant)
        output = self.classifier(x)
        return output
