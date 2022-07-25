from torch import nn
from torch.nn import ZeroPad2d


class DilatedConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spacetime_ndim,
        padding,
        kernel_size,
        dilation,
        activation="ReLU",
    ):
        super(DilatedConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spacetime_ndim = spacetime_ndim
        self.activation = activation

        self.zero_padding = ZeroPad2d(padding)

        if spacetime_ndim == 2:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding='same',
            )
        elif spacetime_ndim == 3:
            self.conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding='same',
            )
        else:
            raise ValueError("spacetime_ndim parameter can only be 2 or 3...")

        self.activation_function = {
            "ReLU": nn.ReLU(),
            "swish": nn.SiLU(),
            "lrel": nn.LeakyReLU(0.1),
        }[self.activation]

    def forward(self, x):
        x = self.zero_padding(x)

        x = self.conv(x)

        x = self.activation_function(x)

        return x