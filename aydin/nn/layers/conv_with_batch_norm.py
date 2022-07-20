from torch import nn


class ConvWithBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spacetime_ndim,
        kernel_size=3,
        normalization=None,  # "batch",
        activation="ReLU",
    ):
        super(ConvWithBatchNorm, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spacetime_ndim = spacetime_ndim
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation

        if spacetime_ndim == 2:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding='same'
            )
            if self.normalization == 'instance':
                self.norm_layer = nn.InstanceNorm2d(out_channels)
            elif self.normalization == 'batch':
                self.norm_layer = nn.BatchNorm2d(out_channels, affine=False)
        else:
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size, padding='same'
            )
            if self.normalization == 'instance':
                self.norm_layer = nn.InstanceNorm3d(out_channels)
            elif self.normalization == 'batch':
                self.norm_layer = nn.BatchNorm3d(out_channels, affine=False)

        if self.activation == 'ReLU':
            self.act_layer = nn.ReLU()
        elif self.activation == 'swish':
            self.act_layer = nn.SiLU()
        elif self.activation == 'lrel':
            self.act_layer = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)

        if self.normalization:
            x = self.norm_layer(x)

        if self.activation:
            x = self.act_layer(x)

        return x
