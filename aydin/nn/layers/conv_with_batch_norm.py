from torch import nn


class ConvWithNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spacetime_ndim,
        kernel_size=3,
        normalization="batch",
        activation="ReLU",
    ):
        super(ConvWithNorm, self).__init__()

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
            self.instance_normalization = nn.InstanceNorm2d(out_channels)
            self.batch_normalization = nn.BatchNorm2d(out_channels, affine=False)
        else:
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size, padding='same'
            )
            self.instance_normalization = nn.InstanceNorm3d(out_channels)
            self.batch_normalization = nn.BatchNorm3d(out_channels, affine=False)

        self.activation_function = {
            "ReLU": nn.ReLU(),
            "swish": nn.SiLU(),
            "lrel": nn.LeakyReLU(0.1),
        }[self.activation]

    def forward(self, x):
        x = self.conv(x)

        if self.normalization == 'instance':
            x = self.instance_normalization(x)
        elif self.normalization == 'batch':
            x = self.batch_normalization(x)

        x = self.activation_function(x)

        return x
