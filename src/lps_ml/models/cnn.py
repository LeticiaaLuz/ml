"""
Module containing a Convolutional Neural Network (CNN) based models.
"""
import typing
import torch
import lightning as L

import lps_ml.models.mlp as lps_mlp


class CNN(L.LightningModule):
    """ CNN with MLP head, compatible with binary or multiclass classification. """

    def __init__(
        self,
        input_shape: typing.Iterable[int],

        conv_n_neurons: typing.List[int],
        conv_activation: typing.Union[torch.nn.Module, typing.Callable] = None,
        conv_pooling: typing.Optional[typing.Callable] = None,
        conv_pooling_size: typing.List[int] = None,
        conv_dropout: float = 0.5,
        batch_norm: typing.Optional[typing.Callable] = None,
        kernel_size: int = 5,
        padding: int = None,

        classification_n_neurons: typing.Union[int, typing.Iterable[int]] = 128,
        n_targets: int = 1,
        classification_dropout: float = 0,
        classification_norm: typing.Optional[typing.Callable] = None,
        classification_hidden_activation: typing.Optional[typing.Callable] = None,
        classification_output_activation: typing.Optional[typing.Callable] = None,
        lr: float = 1e-3,
        loss_fn: typing.Optional[typing.Callable[[], torch.nn.Module]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "conv_activation", "conv_pooling", "batch_norm",
            "classification_hidden_activation", "classification_output_activation", "loss_fn"
        ])

        conv_activation = conv_activation or torch.nn.ReLU
        conv_pooling = conv_pooling or torch.nn.MaxPool2d
        conv_pooling_size = conv_pooling_size or [2, 2]
        batch_norm = batch_norm or torch.nn.BatchNorm2d
        classification_norm = classification_norm or torch.nn.BatchNorm1d

        if loss_fn is None:
            if n_targets == 1:
                loss_fn = torch.nn.BCEWithLogitsLoss
            else:
                loss_fn = torch.nn.CrossEntropyLoss
        self.loss_fn = loss_fn()

        classification_hidden_activation = classification_hidden_activation or conv_activation
        padding = padding or int((kernel_size - 1) / 2)

        if len(input_shape) != 3:
            raise ValueError(f"CNN expects as input an image in the format: \
                                    channel x width x height (current {input_shape})")

        self.input_shape = input_shape
        self.n_targets = n_targets
        self.lr = lr
        self.is_binary = n_targets == 1

        conv_layers = []
        conv_channels = [input_shape[0]] + conv_n_neurons
        for i in range(1, len(conv_channels)):
            conv_layers.append(torch.nn.Conv2d(conv_channels[i-1], conv_channels[i],
                                               kernel_size=kernel_size, padding=padding))
            if batch_norm is not None:
                conv_layers.append(batch_norm(conv_channels[i]))
            if conv_dropout != 0 and i != 0:
                conv_layers.append(torch.nn.Dropout2d(p=conv_dropout))
            conv_layers.append(conv_activation())
            if conv_pooling is not None:
                conv_layers.append(conv_pooling(*conv_pooling_size))
        self.conv_layers = torch.nn.Sequential(*conv_layers)

        test_tensor = torch.rand([1] + list(input_shape), dtype=torch.float32)
        device = next(self.parameters()).device
        test_tensor = test_tensor.to(device)
        self.conv_layers.to(device)
        features = self.to_feature_space(test_tensor)

        self.mlp = lps_mlp.MLP(
            input_shape=features.shape,
            hidden_channels=classification_n_neurons,
            norm_layer=classification_norm,
            n_targets=n_targets,
            activation_layer=classification_hidden_activation,
            activation_output_layer=classification_output_activation,
            dropout=classification_dropout,
            loss_fn=loss_fn
        )

    def to_feature_space(self, x: torch.Tensor) -> torch.Tensor:
        """ Pass the input through the convolutional part of the network."""
        return self.conv_layers(x)

    #pylint: disable=W0221
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_feature_space(x)
        x = self.mlp(x)
        return x

    def _shared_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Shared step for training and validation."""
        x, y = batch
        if self.is_binary:
            y = y.float()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def training_step(self,
                batch: typing.Tuple[torch.Tensor, torch.Tensor],
                _: int = 0) -> torch.Tensor:
        """Executes a single training step."""
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self,
                batch: typing.Tuple[torch.Tensor, torch.Tensor],
                _: int = 0) -> torch.Tensor:
        """Executes a single validation step."""
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self,
                batch: typing.Tuple[torch.Tensor, torch.Tensor],
                _: int = 0) -> torch.Tensor:
        """ Executes a single test step. """
        return self._shared_step(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """ Defines and returns the optimizer used during training. """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
