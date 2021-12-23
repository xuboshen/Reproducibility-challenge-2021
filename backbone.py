import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import random
import matplotlib.pyplot as plt
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# class MixStyle(nn.Module):
#     """MixAll
#     def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='crossdomain'):
#         """
#         Args:
#           p (float): probability of using MixStyle.
#           alpha (float): parameter of the Beta distribution.
#           eps (float): scaling parameter to avoid numerical issues.
#           mix (str): how to mix.
#         """
#         super().__init__()
#         self.p = p
#         self.beta = torch.distributions.Beta(alpha, alpha)
#         self.eps = eps
#         self.alpha = alpha
#         self.mix = mix
#         self._activated = True

#     def __repr__(self):
#         return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

#     def set_activation_status(self, status=True):
#         self._activated = status

#     def update_mix_method(self, mix='crossdomain'):
#         self.mix = mix

#     def forward(self, x):
#         if not self.training or not self._activated:
#             return x

#         if random.random() > self.p:
#             return x

#         B = x.size(0)

#         mu = x.mean(dim=[2, 3], keepdim=True)
#         var = x.var(dim=[2, 3], keepdim=True)
#         sig = (var + self.eps).sqrt()
#         mu, sig = mu.detach(), sig.detach()
#         x_normed = (x-mu) / sig

#         lmda = self.beta.sample((B, 1, 1, 1))
#         lmda1 = self.beta.sample((B, 1, 1, 1))*(1-lmda)
#         lmda = lmda.to(x.device)
#         lmda1 = lmda1.to(x.device)

#         if self.mix == 'random':
#             # random shuffle
#             perm = torch.randperm(B)

#         elif self.mix == 'crossdomain':
#             # split into two halves and swap the order
#             perm = torch.arange(B - 1, -1, -1) # inverse index
#             perm_b, perm_a, perm_c = perm.chunk(3)
#             perm_b = perm_b[torch.randperm(B // 3)]
#             perm_a = perm_a[torch.randperm(B // 3)]
#             perm_c = perm_c[torch.randperm(B // 3)]
#             perm = torch.cat([perm_b, perm_c, perm_a], 0)
#             perm1 = torch.cat([perm_c, perm_a, perm_b], 0)
#         else:
#             raise NotImplementedError

#         mu2, sig2 = mu[perm], sig[perm]
#         mu3, sig3 = mu[perm1], sig[perm1]
#         mu_mix = mu*lmda + mu2 * lmda1 + mu3 * (1 - lmda - lmda1)
#         sig_mix = sig*lmda + sig2 * (lmda1) + sig3 * (1- lmda- lmda1)

#         return x_normed*sig_mix + mu_mix

class MixStyle(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='crossdomain'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='crossdomain'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix
        
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    @property
    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get('_out_features') is None:
            return None
        return self._out_features

class ResNet(Backbone):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_a=0.1,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3']
            print(f'Insert MixStyle after {ms_layers}')
        self.ms_layers = ms_layers

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None 
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x, return_feature=False):
        if return_feature is True:
          mu = x.mean(dim=[2, 3], keepdim=True)
          var = x.var(dim=[2, 3], keepdim=True)
          sig = (var + 1e-6).sqrt()
          mu, sig = mu.detach(), sig.detach()
          mu = mu.reshape(x.shape[0], -1)
          sig = sig.reshape(x.shape[0], -1)
          # print('x.shape:', x.shape)
          # print('concat.shape[0]:',torch.cat((mu, sig), 1).shape)
          return torch.cat((mu, sig), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if 'layer1' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer2(x)
        if 'layer2' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer3(x)
        if 'layer3' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer4(x)
        return x

    def forward(self, x, return_feature=False):
        f = self.featuremaps(x, return_feature)
        # a, b = f[0], f[1]
        if return_feature is True:
          return f,f
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""
def resnet18(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

def resnet34(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])

    return model

def resnet50(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

def resnet101(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model

def resnet152(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])

    return model


"""
Residual networks with mixstyle
"""

def resnet18_ms_l123(pretrained=True, alpha=0.1, **kwargs):
    

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3'],
        ms_a = alpha
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

def resnet18_ms_l12(pretrained=True, **kwargs):
    

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

def resnet18_ms_l1(pretrained=True, **kwargs):
    

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

def resnet18_ms_l3(pretrained=True, **kwargs):
    

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer3']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

def resnet18_ms_l2(pretrained=True, **kwargs):
    

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer2']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

def resnet18_ms_l23(pretrained=True, **kwargs):
    

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer2', 'layer3']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

def resnet50_ms_l123(pretrained=True, alpha=0.1, **kwargs):
    

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3'],
        ms_a=alpha
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

def resnet50_ms_l12(pretrained=True, **kwargs):
    

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

def resnet50_ms_l1(pretrained=True, **kwargs):
    

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model

def resnet101_ms_l123(pretrained=True, **kwargs):
    

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model


def resnet101_ms_l12(pretrained=True, **kwargs):
    

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model

def resnet101_ms_l1(pretrained=True, **kwargs):
    

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=['layer1']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model
