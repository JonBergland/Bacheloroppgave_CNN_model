

import copy
import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import ImageFolder

from data_processing.data_augmentation import DataAugmentation
from data_processing.dataset_loader import DatasetLoader


class DatasetSplitter:
    """
    Splits a dataset into train/test and optionally applies augmentations
    to each split independently.
    """

    def __init__(
        self,
        dataset_root: str,
        img_size: int,
        output_channels: int = 1,
        test_ratio: float = 0.2,
        manual_seed: int = 42,
    ):
        self.dataset_root = dataset_root
        self.img_size = img_size
        self.output_channels = output_channels
        self.test_ratio = test_ratio
        self.manual_seed = manual_seed

        self.dataset_loader = DatasetLoader(
            dataset_root=dataset_root,
            img_size=img_size,
            output_channels=output_channels,
        )

        self.data_augmentation = DataAugmentation(
            img_size=img_size,
            output_channels=output_channels,
        )

        self.train_indices = []
        self.test_indices = []

        self._create_splits()

    def _create_splits(self):
        """Create train/test splits based on test_ratio."""
        n = len(self.dataset_loader.datasets[0])
        indices = np.arange(n)

        rng = np.random.default_rng(self.manual_seed)
        rng.shuffle(indices)

        test_size = int(self.test_ratio * n)
        train_size = n - test_size

        self.train_indices = indices[:train_size].tolist()
        self.test_indices = indices[train_size:].tolist()

    def _apply_augmentations_to_split(
        self,
        indices: list[int],
        augment: bool = False,
    ) -> Dataset:
        """
        Apply augmentations to a dataset split.
        
        Args:
            indices: list of indices for this split
            augment: whether to apply augmentations to this split
        
        Returns:
            Dataset (either ImageFolder or ConcatDataset if augmentations applied)
        """
        datasets = []

        subset_dataset = self._create_subset(indices)
        datasets.append(subset_dataset)

        should_augment = augment

        if should_augment:
            augmentation_transforms = self.data_augmentation.getDataAugmentations(
                horizontal_flip=True,
                vertical_flip=True,
                translation=True,
                blur=True,
                color_jitter=True,
                random_erasing=True,
            )

            for transform in augmentation_transforms:
                aug_subset = self._create_subset(indices, transform=transform)
                datasets.append(aug_subset)

        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def _create_subset(self, indices: list[int], transform=None):
        """Create an copy of the images connected to the indices and use transforms on them"""
        base_dataset = self.dataset_loader.datasets[0]
        subset_dataset = copy.deepcopy(base_dataset)

        if transform is not None:
            subset_dataset.transform = transform

        return Subset(subset_dataset, indices)
        
    def get_train_test_datasets(
        self,
        augment_train: bool = False,
        augment_test: bool = False,
    ) -> tuple[Dataset, Dataset]:
        """
        Get train and test datasets with optional augmentations.
        The test set serves as validation during training.

        Args:
            augment_train: Apply augmentations to train split
            augment_test: Apply augmentations to test split

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train_dataset = self._apply_augmentations_to_split(
            self.train_indices,
            augment=augment_train,
        )
        test_dataset = self._apply_augmentations_to_split(
            self.test_indices,
            augment=augment_test,
        )

        return train_dataset, test_dataset

    def get_split_indices(self) -> tuple[list[int], list[int]]:
        """Return the indices for train and test splits."""
        return self.train_indices, self.test_indices