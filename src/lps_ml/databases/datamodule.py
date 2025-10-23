"""DataModule
"""
import abc
import typing

import lightning

class DataModule(lightning.LightningDataModule):
    """ Basic DataModule """

    def get_sample_shape(self, subset: str = "train") -> typing.List[int]:
        """
        Returns the shape of the dataset samples.
        """
        if not getattr(self, "_has_setup", False):
            self.setup("fit")

        if subset == "train":
            loader = self.train_dataloader()
        elif subset == "val":
            loader = self.val_dataloader()
        elif subset == "test":
            loader = self.test_dataloader()
        elif subset == "predict":
            loader = self.predict_dataloader()
        else:
            raise ValueError(f"invalid subset: {subset}")

        x, _ = next(iter(loader))
        return list(x.shape[1:])

    @abc.abstractmethod
    def get_n_targets(self) -> int:
        """ Return the number of targets in dataset. """
