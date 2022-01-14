import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='tmp')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpus', action='store_true')
    parser.add_argument('--dist', action='store_true')

    parser.add_argument('--celeb_csv', type=str, default='./repo/celeba.csv')

    # data
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--train_val_max', type=int, default=15000)
    parser.add_argument('--train_val_min', type=int, default=10000)
    parser.add_argument('--majority_frac', type=float, default=0.5)
    parser.add_argument('--val_frac', type=float, default=0.5)
    parser.add_argument('--dset_num', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=84)
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--batch_num', type=int, default=16)
    parser.add_argument('--batch_thd', type=float, default=0)

    # train
    parser.add_argument('--epoches', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--train_frac', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_interval', type=float, default=200)
    parser.add_argument('--encode', type=str, default='ours_mm')
    parser.add_argument('--middle', type=str, default='pass')
    parser.add_argument('--decode', type=str, default='pass')

    # model
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--kpop', type=int, default=5)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--word_embed_size', type=int, default=128)
    parser.add_argument('--gru_hidden_dim', type=int, default=128)

    opt = parser.parse_args()

    return opt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx]


class Conv2dCustom(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kernel = kwargs.get("kernel", 3)
        self.stride = kwargs.get("stride", 1)
        self.in_feat = kwargs.get("in_feat", 3)
        self.out_feat = kwargs.get("out_feat", 3)
        self.batch_norm = kwargs.get("batch_norm", True)
        self.transpose = kwargs.get("transpose", False)
        conv = torch.nn.Conv2d(self.in_feat, self.out_feat, self.kernel, stride=self.stride) if not self.transpose else torch.nn.ConvTranspose2d(
            self.in_feat, self.out_feat, self.kernel, stride=self.stride)
        if self.batch_norm:
            seq = torch.nn.Sequential(*[
                conv,
                torch.nn.BatchNorm2d(self.out_feat),
                torch.nn.LeakyReLU(True)
            ])
        else:
            seq = torch.nn.Sequential(*[
                conv,
                torch.nn.LeakyReLU(True)
            ])
        self.add_module("conv_unit", seq)

    def forward(self, inputs):
        x = inputs
        return self.conv_unit(x)


class Fourier3dCustom(torch.nn.Module):
    def __init__(self, **kwargs):
        self.split = kwargs.get("split", False)
        super().__init__()

    def forward(self, inputs):
        x = inputs
        x = self.fourier_split(x) if self.split else self.fourier_merge(x)
        return x

    def fourier_split(self, inputs):
        ffted = torch.fft.fftn(inputs, dim=(-3, -2, -1))
        return ffted.real, ffted.imag

    def fourier_merge(self, inputs):
        real, imag = inputs[0], inputs[1]
        precomp = torch.stack((real, imag), dim=-1)
        complx = torch.view_as_complex(precomp)
        iffted = torch.fft.ifftn(complx, dim=(-3, -2, -1))
        return iffted


class LeftModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        left_config = kwargs.get("left_config", None)
        seq = torch.nn.ModuleList()
        for i in range(len(left_config)-1):
            key = "c" + f"{i}"
            config = left_config.get(key)
            conv_unit = Conv2dCustom(**config)
            seq.append(conv_unit)
        self.add_module("conv_blocks", torch.nn.Sequential(*seq))
        self.add_module("remaining_block", torch.nn.Sequential(*[
            torch.nn.BatchNorm2d(left_config.get("c9").get("in_feat")),
            torch.nn.ConvTranspose2d(left_config.get("c9").get("in_feat"),
                                     left_config.get(
                        "c9").get("out_feat"),
                kernel_size=left_config.get(
                "c9").get("kernel"),
                stride=left_config.get("c9").get("stride")),
            torch.nn.Tanh()
        ]))

    def forward(self, inputs):
        x = inputs
        x = self.conv_blocks(x)
        x = self.remaining_block(x)
        return x


class RightModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        right_config = kwargs.get("right_config", None)
        seq = torch.nn.ModuleDict()
        conv = torch.nn.ModuleList()
        for key in list(right_config.keys()):
            config = right_config.get(key)
            if key in ["begin", "end"]:
                setattr(seq, key, Fourier3dCustom(**config))
            elif key.startswith("c"):
                conv.append(Conv2dCustom(**config))
            elif key == "up":
                setattr(seq, key, torch.nn.AdaptiveAvgPool3d(**config))
            elif key == "down":
                setattr(seq, key, torch.nn.Upsample(**config))
            elif key == "drop":
                setattr(seq, key, torch.nn.Dropout3d(**config))
            elif key == "bnorm":
                setattr(seq, key, torch.nn.BatchNorm3d(**config))
        self.add_module("conv_block", torch.nn.Sequential(*conv))
        self.seq = seq

    def forward(self, inputs):
        x = inputs
        r, i = getattr(self.seq, "begin")(x)
        r, i = list(map(lambda inp: getattr(self.seq, "up")(inp), (r, i)))
        r, i = list(map(lambda inp: self.conv_block(inp), (r, i)))
        r, i = list(map(lambda inp: getattr(self.seq, "drop")(inp), (r, i)))
        r, i = list(map(lambda inp: getattr(self.seq, "down")(inp), (r, i)))
        x = getattr(self.seq, "end")([r, i])
        # x = getattr(self.seq, "bnorm")(x)
        return x


if __name__ == '__main__':
    opt = get_parser()

    data = torch.ones(*[128] * 3)[None,
                                 None, ...].expand(opt.batch_size, *[-1]*4)  # (B,C,D0,D1,D2)
    diffed = [data]
    dt = 1/10
    for n in range(1, 11):
        diff = torch.ones_like(
            data) * dt * n * torch.distributions.Uniform(0., 1.).rsample(data.size())
        diffed.append(data - diff)
    data = torch.stack(diffed, dim=1)  # (B,T,C,D0,D1,D2)

    left_config = dict(
        c0=dict(in_feat=1, out_feat=128, kernel=7,
                stride=1, batch_norm=True, transpose=False),
        c1=dict(in_feat=128, out_feat=64, kernel=7,
                stride=1, batch_norm=True, transpose=False),
        c2=dict(in_feat=64, out_feat=128, kernel=7,
                stride=2, batch_norm=True, transpose=False),
        c3=dict(in_feat=128, out_feat=64, kernel=7,
                stride=2, batch_norm=True, transpose=False),
        c4=dict(in_feat=64, out_feat=64, kernel=7,
                stride=2, batch_norm=True, transpose=True),
        c5=dict(in_feat=64, out_feat=32, kernel=7,
                stride=2, batch_norm=True, transpose=True),
        c6=dict(in_feat=32, out_feat=128, kernel=7,
                stride=1, batch_norm=True, transpose=False),
        c7=dict(in_feat=128, out_feat=64, kernel=7,
                stride=1, batch_norm=True, transpose=False),
        c8=dict(in_feat=64, out_feat=1, kernel=7, stride=1,
                batch_norm=True, transpose=False),
        c9=dict(in_feat=1, out_feat=1, kernel=7, stride=1,
                batch_norm=True, transpose=True),
    )
    right_config = dict(
        begin=dict(split=True),
        up=dict(output_size=64),
        c0=dict(in_feat=1, out_feat=128, kernel=7, stride=1,
                batch_norm=False, transpose=False),
        c1=dict(in_feat=128, out_feat=1, kernel=7,
                stride=1, batch_norm=False, transpose=True),
        c2=dict(in_feat=1, out_feat=128, kernel=7, stride=1,
                batch_norm=False, transpose=False),
        c3=dict(in_feat=128, out_feat=1, kernel=7,
                stride=1, batch_norm=False, transpose=True),
        drop=dict(p=0.3),
        down=dict(size=64, mode='trilinear', align_corners=False),
        end=dict(split=False),
        bnorm=dict(num_features=1))

    l_model = LeftModel(left_config=left_config)
    r_model = RightModel(right_config=right_config)

    dataset = Dataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=True, collate_fn=None)

    print(l_model(next(iter(dataloader))[:, 0, ..., 0]).size())
    print(r_model(next(iter(dataloader))[:, 0, ..., 0]).size())