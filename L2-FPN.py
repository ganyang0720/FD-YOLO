import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CA', 'multiply', 'Add']


class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()

    def forward(self, x):
        input1, input2 = x[0], x[1]
        x = input1 + input2
        return x


class multiply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[0] * x[1]
        return x


class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7, flag=True):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()
        self.flag = flag  # ✅ 控制是否乘以原输入

    def forward(self, x):
        b, c, h, w = x.size()

        # 高度方向注意力
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)         # [B, C, H]
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)   # [B, C, H, 1]

        # 宽度方向注意力
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)         # [B, C, W]
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)   # [B, C, 1, W]

        out = x_h * x_w  # 注意力图 [B, C, H, W]
        return x * out if self.flag else out


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1)
        self.group_norm1 = nn.GroupNorm(32, in_chan)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1)
        self.group_norm2 = nn.GroupNorm(32, out_chan)
        nn.init.xavier_uniform_(self.conv_atten.weight)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        atten = self.sigmoid(self.group_norm1(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.group_norm2(self.conv(x))
        return feat


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # Model
    mobilenet_v3 = FeatureSelectionModule(64, 64)

    out = mobilenet_v3(image)
    print(out.size())