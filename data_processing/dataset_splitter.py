import copy
import inspect
import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

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
        val_ratio: float = 0.15,
        manual_seed: int = 42,
        use_kfold: bool = False,
        n_splits: int = 5,
        stratified_kfold: bool = True,
        dataset_is_preprocessed: bool = True,
    ):
        self.dataset_root = dataset_root
        self.img_size = img_size
        self.output_channels = output_channels
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.manual_seed = manual_seed
        self.use_kfold = use_kfold
        self.n_splits = n_splits
        self.stratified_kfold = stratified_kfold
        self.dataset_is_preprocessed = dataset_is_preprocessed

        self.dataset_loader = DatasetLoader(
            dataset_root=dataset_root,
            img_size=img_size,
            output_channels=output_channels,
        )

        self.data_augmentation = DataAugmentation(
            img_size=img_size,
            output_channels=output_channels,
        )

        self.base_dataset = self.dataset_loader.datasets[0]
        self.classes = self.base_dataset.classes
        self.targets = np.array(self.base_dataset.targets)

        self.train_indices: list[int] = []
        self.val_indices: list[int] = []
        self.test_indices: list[int] = []
        self.trainval_indices: list[int] = []
        self.fold_splits: list[tuple[list[int], list[int]]] = []
        self.current_fold = 0

        self._initialize_splits()

    def _initialize_splits(self):
        if self.use_kfold:
            self._initialize_holdout_and_folds()
        else:
            self._initialize_fixed_split()

    def _initialize_fixed_split(self):
        n = len(self.base_dataset)
        indices = np.arange(n)
        targets = self.targets

        test_ratio = float(self.test_ratio)
        if test_ratio <= 0 or test_ratio >= 1:
            raise ValueError("test_ratio must be in (0, 1)")
        if self.val_ratio <= 0 or self.val_ratio >= 1:
            raise ValueError("val_ratio must be in (0, 1)")
        if self.val_ratio + test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be less than 1")

        stratify_all = targets if self.stratified_kfold else None
        trainval_indices, test_indices = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=self.manual_seed,
            shuffle=True,
            stratify=stratify_all,
        )

        val_fraction_of_trainval = self.val_ratio / (1.0 - test_ratio)
        stratify_trainval = targets[trainval_indices] if self.stratified_kfold else None
        train_indices, val_indices = train_test_split(
            trainval_indices,
            test_size=val_fraction_of_trainval,
            random_state=self.manual_seed,
            shuffle=True,
            stratify=stratify_trainval,
        )

        self.train_indices = train_indices.tolist()
        self.val_indices = val_indices.tolist()
        self.test_indices = test_indices.tolist()

    def _initialize_holdout_and_folds(self):
        n = len(self.base_dataset)
        indices = np.arange(n)

        rng = np.random.default_rng(self.manual_seed)
        rng.shuffle(indices)

        test_size = int(self.test_ratio * n)
        test_size = max(1, min(test_size, n - 1))

        self.test_indices = indices[:test_size].tolist()
        self.trainval_indices = indices[test_size:].tolist()

        if len(self.trainval_indices) < self.n_splits:
            raise ValueError(
                f"n_splits={self.n_splits} is too large for trainval size={len(self.trainval_indices)}"
            )

        trainval_array = np.array(self.trainval_indices)
        trainval_targets = self.targets[trainval_array]

        if self.stratified_kfold:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.manual_seed,
            )
            fold_iter = splitter.split(trainval_array, trainval_targets)
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.manual_seed,
            )
            fold_iter = splitter.split(trainval_array)

        self.fold_splits = []
        for train_rel, val_rel in fold_iter:
            fold_train = trainval_array[train_rel].tolist()
            fold_val = trainval_array[val_rel].tolist()
            self.fold_splits.append((fold_train, fold_val))

        self.set_fold(0)

    def _apply_augmentations_to_split(
        self,
        indices: list[int],
        augment: bool = False,
        transform_options: dict | None = None,
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

        if augment:
            default_options = {
                "horizontal_flip": False,
                "vertical_flip": False,
                "translation": False,
                "blur": False,
                "random_erasing": False,
                "noise_light": False,
                "noise_medium": False,
                "noise_strong": False,
                "scale": False,
            }
            options = default_options if transform_options is None else {**default_options, **transform_options}
            options["preprocessed_input"] = self.dataset_is_preprocessed

            augmentation_transforms = self.data_augmentation.getDataAugmentations(**options)

            for transform in augmentation_transforms:
                aug_subset = self._create_subset(indices, transform=transform)
                datasets.append(aug_subset)

        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def _create_subset(self, indices: list[int], transform=None):
        """Create an copy of the images connected to the indices and use transforms on them"""
        base_dataset = self.base_dataset
        subset_dataset = copy.deepcopy(base_dataset)

        if transform is not None:
            subset_dataset.transform = transform

        return Subset(subset_dataset, indices)

    def build_split_dataset(
        self,
        indices: list[int],
        augment: bool = False,
        transform_options: dict | None = None,
    ) -> Dataset:
        return self._apply_augmentations_to_split(
            indices,
            augment=augment,
            transform_options=transform_options,
        )

    def set_fold(self, fold_index: int):
        if not self.fold_splits:
            raise RuntimeError("No fold splits are available. Set use_kfold=True to use folds.")
        if fold_index < 0 or fold_index >= len(self.fold_splits):
            raise IndexError(f"fold_index must be in [0, {len(self.fold_splits) - 1}]")

        self.current_fold = fold_index
        self.train_indices, self.val_indices = self.fold_splits[fold_index]

    def iter_folds(self):
        for fold_index, (train_indices, val_indices) in enumerate(self.fold_splits):
            yield fold_index, train_indices, val_indices

    def fold_count(self):
        return len(self.fold_splits)

    def get_train_test_datasets(
        self,
        augment_train: bool = False,
        augment_test: bool = False,
        train_transform_options: dict | None = None,
        test_transform_options: dict | None = None,
    ) -> tuple[Dataset, Dataset]:
        """
        Get train and test datasets with optional augmentations.
        The test set serves as validation during training.

        Args:
            augment_train: Apply augmentations to train split
            augment_test: Apply augmentations to test split

        Returns:
            Tuple of (train_dataset, test_dataset)

        Exampel code:
        train_ds, test_ds = splitter.get_train_test_datasets(
            augment_train=True,
            augment_test=True,
            train_transform_options={
                "horizontal_flip": False,
                "vertical_flip": True,
                "translation": True,
                "blur": False,
                "random_erasing": False,
                "noise": False,
                "scale": True,
            },
            test_transform_options={
                "horizontal_flip": False,
                "vertical_flip": False,
                "translation": False,
                "blur": False,
                "random_erasing": False,
                "noise": True,
                "scale": False,
            },
        )
        """
        if self.use_kfold and not self.train_indices:
            self.set_fold(0)

        train_dataset = self._apply_augmentations_to_split(
            self.train_indices,
            augment=augment_train,
            transform_options=train_transform_options,
        )
        test_dataset = self._apply_augmentations_to_split(
            self.test_indices,
            augment=augment_test,
            transform_options=test_transform_options,
        )

        return train_dataset, test_dataset

    def get_train_val_test_datasets(
        self,
        augment_train: bool = False,
        augment_val: bool = False,
        augment_test: bool = False,
        train_transform_options: dict | None = None,
        val_transform_options: dict | None = None,
        test_transform_options: dict | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        if self.use_kfold and not self.train_indices:
            self.set_fold(0)

        train_dataset = self._apply_augmentations_to_split(
            self.train_indices,
            augment=augment_train,
            transform_options=train_transform_options,
        )
        val_dataset = self._apply_augmentations_to_split(
            self.val_indices,
            augment=augment_val,
            transform_options=val_transform_options,
        )
        test_dataset = self._apply_augmentations_to_split(
            self.test_indices,
            augment=augment_test,
            transform_options=test_transform_options,
        )

        return train_dataset, val_dataset, test_dataset

    def get_split_indices(self) -> tuple[list[int], list[int]]:
        """Return the indices for train and test splits (backward-compatible)."""
        return self.train_indices, self.test_indices

    def get_train_val_test_indices(self) -> tuple[list[int], list[int], list[int]]:
        """Return the indices for train, val and test splits."""
        return self.train_indices, self.val_indices, self.test_indices