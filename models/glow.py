import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import transforms


class Squeeze(transforms.Transform):
    bijective = True
    event_dim = 3
    def __init__(self, factor):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(factor, int)
        self._factor = factor

    def _call(self, input):
        factor = self._factor
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x

    def _inverse(self, input):
        factor = self._factor
        factor2 = factor ** 2
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

    def log_abs_det_jacobian(self, x, y):
        return 0



class InvertibleConv1x1(transforms.Transform):
    bijective = True
    event_dim = 3
    def __init__(self, module):
        super(InvertibleConv1x1, self).__init__()
        self.module = module

    def _call(self, x):
        W = self.module.forward_kernel()
        return F.conv2d(x, W)

    def _inverse(self, x):
        W = self.module.inverse_kernel()
        return F.conv2d(x, W)

    def log_abs_det_jacobian(self, x, y):
        ndims = np.prod(x.shape[2], x.shape[3])
        return ndims * torch.log(torch.abs(torch.det(self.module.W)))


def split_half(x):
    s = x.shape[1] // 2
    return x[:, :s, ...], x[:, s:, ...]


class ChannelCoupling(transforms.Transform):
    bijective = True
    event_dim = 3

    def __init__(self, network):
        super(ChannelCoupling, self).__init__()
        self._nn = network

    def _call(self, x):
        x1, x2 = split_half(x)
        offset, scale = split_half(self._nn(x1))
        scale = torch.sigmoid(scale + 2.)
        y2 = x2 * scale + offset
        y = torch.cat([x1, y2], dim=1)
        return y

    def _inverse(self, x):
        x1, x2 = split_half(x)
        offset, scale = split_half(self._nn(x1))
        scale = torch.sigmoid(scale + 2.)
        y2 = (x2 - offset) / scale
        y = torch.cat([x1, y2], dim=1)
        return y

    def log_abs_det_jacobian(self, x, y):
        x1, x2 = split_half(x)
        _, scale = split_half(self._nn(x1))
        return torch.sum(torch.log(scale), dim=[1, 2, 3])


def _padding(k, s):
    return ((k - 1) * s + 1) // 2


def _conv(in_dim, out_dim, kernel_size, zero=False):
    conv = nn.Conv2d(in_dim,
                     out_dim,
                     kernel_size=kernel_size,
                     padding=_padding(kernel_size, 1))
    if zero:
        nn.init.constant_(conv.weight, 0.)
    else:
        nn.init.normal_(conv.weight, 0., 0.02)
    nn.init.constant_(conv.bias, 0.)
    return conv



class InvertibleConv1x1Matrix(nn.Module):
    def __init__(self, c):
        super(InvertibleConv1x1Matrix, self).__init__()
        self._c = c
        self.w_init = torch.Tensor(np.random.randn(self._c, self._c)).qr()[0]
        # w_init = np.linalg.qr(
        #     np.random.randn(self._c, self._c))[0].astype(np.float32)
        self.weight = nn.Parameter(self.w_init)

    def forward_kernel(self):
        return self.weight.view((self._c, self._c, 1, 1))

    def inverse_kernel(self):
        return self.weight.inverse().view((self._c, self._c, 1, 1))


def TanhTransform():
    return transforms.ComposeTransform(
        [transforms.AffineTransform(loc=0., scale=2.),
         transforms.SigmoidTransform(),
         transforms.AffineTransform(loc=-1., scale=2.)])


class Glow(nn.Module):
    def __init__(self, n, c, h, squeeze):
        super(Glow, self).__init__()
        self._n_couple = 0
        self._n_conv1x1 = 0
        self._n, self._c, self._h = n, c, h

        t = []
        #t.append(TanhTransform().inv)
        if squeeze > 1:
            t.append(Squeeze(squeeze))
            self._c = self._c * (squeeze ** 2)
        for _ in range(self._n):
            t.extend([
                InvertibleConv1x1(self._conv1x1_network()),
                ChannelCoupling(self._coupling_network())
            ])
        t.append(InvertibleConv1x1(self._conv1x1_network()))
        if squeeze > 1:
            t.append(Squeeze(squeeze).inv)
            self._c = self._c // (squeeze ** 2)
        #t.append(TanhTransform())
        self.transform = transforms.ComposeTransform(t)

    def _coupling_network(self):
        c, h = self._c, self._h
        module = nn.Sequential(
            _conv(c//2, h, kernel_size=3),
            nn.BatchNorm2d(h),
            nn.LeakyReLU(inplace=False),
            _conv(h, h, kernel_size=1),
            nn.BatchNorm2d(h),
            nn.LeakyReLU(inplace=False),
            _conv(h, c, kernel_size=3, zero=True),
            # _conv(c//2, h, kernel_size=5),
            # #nn.InstanceNorm2d(h),
            # nn.BatchNorm2d(h),
            # nn.LeakyReLU(inplace=False),
            # _conv(h, h, kernel_size=3),
            # # nn.InstanceNorm2d(h),
            # nn.BatchNorm2d(h),
            # nn.LeakyReLU(inplace=False),
            # _conv(h, h, kernel_size=3),
            # # nn.InstanceNorm2d(h),
            # nn.BatchNorm2d(h),
            # nn.LeakyReLU(inplace=False),
            # _conv(h, c, kernel_size=5, zero=True),
        )
        self.add_module("couple_{}".format(self._n_couple), module)
        self._n_couple += 1
        return module

    def _conv1x1_network(self):
        module = InvertibleConv1x1Matrix(self._c)
        self.add_module("conv1x1_{}".format(self._n_conv1x1), module)
        self._n_conv1x1 += 1
        return module



if __name__ == "__main__":

    loss = torch.nn.L1Loss()

    # x = torch.rand((1, 3, 256, 256)) * 2 - 1
    glow = Glow(32, 3, 16, squeeze=4)

    if torch.cuda.is_available():
        print("cuda is available")
        glow.cuda()
    else:
        print("cuda is not available")

    transform = glow.transform.inv
    # z = transform(x)
    # x2 = transform.inv(z)
    # print(loss(x, x2))

    import torchvision.transforms as img_transforms
    from torch.utils.data import DataLoader
    from PIL import Image
    from datasets import ImageDataset

    transforms_ = [ img_transforms.Resize(int(256*1.12), Image.BICUBIC),
                    img_transforms.RandomCrop(256),
                    img_transforms.RandomHorizontalFlip(),
                    img_transforms.ToTensor(),
                    img_transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    dataloader = DataLoader(
        ImageDataset('datasets/facades',
                     transforms_=transforms_, unaligned=True),
        batch_size=1, shuffle=False, num_workers=0)

    for d in dataloader:
        x = d['A'] # * 0.99
        z = transform(x) # * 0.99
        # z = torch.max(z, torch.Tensor(np.array(0.99)))
        # z = torch.min(z, torch.Tensor(np.array(-0.99)))
        x2 = transform.inv(z)
        print(loss(x, x2))
        break
