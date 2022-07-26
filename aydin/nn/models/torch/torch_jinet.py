from collections import OrderedDict
from itertools import chain

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from aydin.nn.layers.dilated_conv import DilatedConv
from aydin.nn.pytorch.optimizers.esadam import ESAdam
from aydin.util.log.log import lprint


class JINetModel(nn.Module):
    def __init__(
        self,
        spacetime_ndim,
        nb_out_channels: int = 1,
        kernel_sizes=None,
        num_features=None,
        nb_dense_layers: int = 3,
        nb_channels: int = None,
        final_relu: bool = False,
    ):
        super(JINetModel, self).__init__()

        self.spacetime_ndim = spacetime_ndim
        self.nb_out_channels = nb_out_channels
        self._kernel_sizes = kernel_sizes
        self._num_features = num_features
        self.nb_dense_layers = nb_dense_layers
        self.nb_channels = nb_channels
        self.final_relu = final_relu

        if len(self.kernel_sizes) != len(self.num_features):
            raise ValueError("Number of kernel sizes and features does not match.")

        self.dilated_conv_functions = []
        current_receptive_field_radius = 0
        for scale_index in range(len(self.kernel_sizes)):
            # Get kernel size and number of features:
            kernel_size = self.kernel_sizes[scale_index]

            # radius and dilation:
            radius = (kernel_size - 1) // 2
            dilation = 1 + current_receptive_field_radius

            self.dilated_conv_functions.append(
                DilatedConv(
                    1 if scale_index == 0 else self.num_features[scale_index - 1],
                    self.num_features[scale_index],
                    self.spacetime_ndim,
                    padding=dilation * radius,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation="lrel",
                )
            )

            # update receptive field radius
            current_receptive_field_radius += dilation * radius

        if spacetime_ndim == 2:
            self.conv = nn.Conv2d
        elif spacetime_ndim == 3:
            self.conv = nn.Conv3d
        else:
            raise ValueError("spacetime_ndim can not be anything other than 2 or 3...")

        if self.nb_channels is None:
            self.nb_channels = sum(self.num_features)  # * 2

        nb_out = self.nb_channels
        self.kernel_one_conv_functions = []
        for index in range(self.nb_dense_layers):
            nb_in = nb_out
            nb_out = (
                self.nb_out_channels
                if index == (self.nb_dense_layers - 1)
                else self.nb_channels
            )
            print(index, nb_in, nb_out)

            self.kernel_one_conv_functions.append(
                self.conv(
                    in_channels=nb_in,
                    out_channels=nb_out,
                    kernel_size=(1,) * spacetime_ndim,
                )
            )

        self.final_kernel_one_conv = self.conv(
            in_channels=self.nb_channels,
            out_channels=1,
            kernel_size=(1,) * spacetime_ndim,
        )

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    @property
    def kernel_sizes(self):
        if self._kernel_sizes is None:
            if self.spacetime_ndim == 2:
                self._kernel_sizes = [7, 5, 3, 3, 3, 3, 3, 3]
            elif self.spacetime_ndim == 3:
                self._kernel_sizes = [7, 5, 3, 3]

        return self._kernel_sizes

    @property
    def num_features(self):
        if self._num_features is None:
            if self.spacetime_ndim == 2:
                self._num_features = [64, 32, 16, 8, 4, 2, 1, 1]
            elif self.spacetime_ndim == 3:
                self._num_features = [10, 8, 4, 2]

        return self._num_features

    def forward(self, x):
        dilated_conv_list = []

        # Calculate dilated convolutions
        for index in range(len(self.kernel_sizes)):
            x = self.dilated_conv_functions[index](x)
            dilated_conv_list.append(x)
            print(x.shape)

        # Concat the results
        x = torch.cat(dilated_conv_list, dim=1)
        print(f"after cat: {x.shape}")

        # First kernel size one conv
        x = self.kernel_one_conv_functions[0](x)
        print(f"after first kernel one conv: {x.shape}")
        x = self.lrelu(x)
        y = x

        # Rest of the kernel size one convolutions
        for index in range(1, self.nb_dense_layers):
            x = self.kernel_one_conv_functions[index](x)
            x = self.lrelu(x)
            y += x

        # Final kernel size one convolution
        y = self.final_kernel_one_conv(y)

        # Final ReLU
        if self.final_relu:
            y = self.relu(y)

        return y


def n2t_jinet_train_loop(
    input_images,
    target_images,
    model: JINetModel,
    nb_epochs: int = 1024,
    learning_rate=0.01,
    training_noise=0.001,
    l2_weight_regularization=1e-9,
    patience=128,
    patience_epsilon=0.0,
    reduce_lr_factor=0.5,
    reload_best_model_period=1024,
    best_val_loss_value=None,
):
    writer = SummaryWriter()

    reduce_lr_patience = patience // 2

    optimizer = ESAdam(
        chain(model.parameters()),
        lr=learning_rate,
        start_noise_level=training_noise,
        weight_decay=l2_weight_regularization,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        factor=reduce_lr_factor,
        verbose=True,
        patience=reduce_lr_patience,
    )

    def loss_function(u, v):
        return torch.abs(u - v)

    for epoch in range(nb_epochs):
        train_loss_value = 0
        validation_loss_value = 0
        iteration = 0
        for i, (input_image, target_image) in enumerate(
            zip([input_images], [target_images])
        ):
            optimizer.zero_grad()

            model.train()

            translated_image = model(input_image)

            translation_loss = loss_function(translated_image, target_image)

            translation_loss_value = translation_loss.mean()

            translation_loss_value.backward()

            optimizer.step()

            train_loss_value += translation_loss_value.item()
            iteration += 1

            # Validation:
            with torch.no_grad():
                model.eval()

                translated_image = model(input_image)

                translation_loss = loss_function(translated_image, target_image)

                translation_loss_value = translation_loss.mean().cpu().item()

                validation_loss_value += translation_loss_value
                iteration += 1

        train_loss_value /= iteration
        lprint(f"Training loss value: {train_loss_value}")

        validation_loss_value /= iteration
        lprint(f"Validation loss value: {validation_loss_value}")

        writer.add_scalar("Loss/train", train_loss_value, epoch)
        writer.add_scalar("Loss/valid", validation_loss_value, epoch)

        scheduler.step(validation_loss_value)

        if validation_loss_value < best_val_loss_value:
            lprint("## New best val loss!")
            if validation_loss_value < best_val_loss_value - patience_epsilon:
                lprint("## Good enough to reset patience!")
                patience_counter = 0

            best_val_loss_value = validation_loss_value

            best_model_state_dict = OrderedDict(
                {k: v.to('cpu') for k, v in model.state_dict().items()}
            )
        else:
            if epoch % max(1, reload_best_model_period) == 0 and best_model_state_dict:
                lprint("Reloading best models to date!")
                model.load_state_dict(best_model_state_dict)

            if patience_counter > patience:
                lprint("Early stopping!")
                break

            lprint(
                f"No improvement of validation losses, patience = {patience_counter}/{patience}"
            )
            patience_counter += 1

        lprint(f"## Best val loss: {best_val_loss_value}")

    writer.flush()
    writer.close()


# def n2s_jinet_train_loop():
#     writer = SummaryWriter()
#
#     optimizer = ESAdam(
#         chain(model.parameters()),
#         lr=learning_rate,
#         start_noise_level=training_noise,
#         weight_decay=l2_weight_regularisation,
#     )
#
#     writer.flush()
#     writer.close()
