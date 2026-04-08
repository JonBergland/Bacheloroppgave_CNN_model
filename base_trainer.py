import os
import copy
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision.transforms import v2

from data_processing.data_augmentation import DataAugmentation

class BaseTrainer:
    def __init__(self, 
                 dataset_root: str, 
                 model_name: str,
                 epochs: int = 5,
                 lr_rate: float = 0.01,
                 batch_size: int = 32,
                 img_size: int = 64, 
                 manual_seed: int = 42,
                 save_path: str | None = None,
                 output_channels: int = 1,
                 use_kfold: bool = False,
                 n_splits: int = 5,
                 test_ratio: float = 0.15,
                 stratified_kfold: bool = True,
                 num_workers: int = 1,
                 augment_train_split: bool = False,
                 augment_test_split: bool = False,
                 dataset_is_preprocessed: bool = True):

        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.manual_seed = manual_seed
        self.output_channels = output_channels
        self.use_kfold = use_kfold
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        self.stratified_kfold = stratified_kfold
        self.num_workers = num_workers
        self.dataset_root = dataset_root
        self.augment_train_split = augment_train_split
        self.augment_test_split = augment_test_split
        self.dataset_is_preprocessed = dataset_is_preprocessed

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device type: {self.device_type}")
        self.device = torch.device(self.device_type)

        self.generator = torch.Generator().manual_seed(manual_seed)

        self.data_augmentation = DataAugmentation(img_size=img_size, output_channels=output_channels)

        self.base_dataset = torchvision.datasets.ImageFolder(
            root=dataset_root,
            transform=self.data_augmentation.getBaseTransform(preprocessed_input=dataset_is_preprocessed),
        )
        self.dataset = self.base_dataset

        self.classes = self.base_dataset.classes
        self.targets = np.array(self.base_dataset.targets)

        self.trainloader: DataLoader | None = None
        self.valloader: DataLoader | None = None
        self.testloader: DataLoader | None = None

        self.train_indices: list[int] = []
        self.val_indices: list[int] = []
        self.test_indices: list[int] = []
        self.trainval_indices: list[int] = []
        self.fold_splits: list[tuple[list[int], list[int]]] = []
        self.current_fold = 0

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracy: float | None = None
        self.start_epoch = 0

        if save_path is None:
            self.save_path = os.path.join(os.getcwd(), model_name)
        else:
            if os.path.isdir(save_path) or str(save_path).endswith(os.sep):
                os.makedirs(save_path, exist_ok=True)
                self.save_path = os.path.join(save_path, model_name)
            else:
                parent = os.path.dirname(save_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self.save_path = save_path

        self._initialize_data_splits()

    def _initialize_data_splits(self):
        if self.use_kfold:
            self._initialize_holdout_and_folds()
        else:
            self._initialize_fixed_split()

    def make_train_transform(self):
        ops = [v2.Resize((self.img_size, self.img_size))]
        ops.extend([
            v2.Grayscale(num_output_channels=self.output_channels),
            v2.ToTensor(),
            v2.Normalize(mean=[0.449], std=[0.226]),
        ])
        return v2.Compose(ops)

    def make_eval_transform(self):
        return v2.Compose([
            v2.Resize((self.img_size, self.img_size)),
            v2.Grayscale(num_output_channels=self.output_channels),
            v2.ToTensor(),
            v2.Normalize(mean=[0.449], std=[0.226]),
        ])

    def _initialize_fixed_split(self):
        n = len(self.base_dataset)
        indices = torch.randperm(n, generator=self.generator).tolist()

        test_size = int(self.test_ratio * n)
        test_size = max(1, min(test_size, n - 1))

        self.test_indices = indices[:test_size]
        self.train_indices = indices[test_size:]

        train_dataset = self._build_split_dataset(self.train_indices, augment=self.augment_train_split)
        test_dataset = self._build_split_dataset(self.test_indices, augment=self.augment_test_split)

        self.trainloader = self._build_loader(train_dataset, shuffle=True)
        self.valloader = None
        self.testloader = self._build_loader(test_dataset, shuffle=False)

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
        test_dataset = self._build_split_dataset(self.test_indices, augment=self.augment_test_split)
        self.testloader = self._build_loader(test_dataset, shuffle=False)

    def _build_split_dataset(self, indices: list[int], augment: bool = False) -> Dataset:
        datasets: list[Dataset] = [Subset(self.base_dataset, indices)]

        if augment:
            augmentation_transforms = self.data_augmentation.getDataAugmentations(
                horizontal_flip=True,
                vertical_flip=True,
                translation=True,
                blur=True,
                color_jitter=True,
                random_erasing=True,
                preprocessed_input=self.dataset_is_preprocessed,
            )

            for transform in augmentation_transforms:
                augmented_dataset = copy.deepcopy(self.base_dataset)
                augmented_dataset.transform = transform
                datasets.append(Subset(augmented_dataset, indices))

        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def _build_loader(self, dataset: Dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.device_type == "cuda",
            persistent_workers=self.num_workers > 0,
        )

    def set_fold(self, fold_index: int):
        if not self.fold_splits:
            raise RuntimeError("No fold splits are available. Set use_kfold=True to use folds.")
        if fold_index < 0 or fold_index >= len(self.fold_splits):
            raise IndexError(f"fold_index must be in [0, {len(self.fold_splits) - 1}]")

        self.current_fold = fold_index
        self.train_indices, self.val_indices = self.fold_splits[fold_index]
        train_dataset = self._build_split_dataset(self.train_indices, augment=self.augment_train_split)
        val_dataset = self._build_split_dataset(self.val_indices, augment=self.augment_test_split)
        self.trainloader = self._build_loader(train_dataset, shuffle=True)
        self.valloader = self._build_loader(val_dataset, shuffle=False)

    def iter_folds(self) -> Iterator[tuple[int, list[int], list[int]]]:
        for fold_index, (train_indices, val_indices) in enumerate(self.fold_splits):
            yield fold_index, train_indices, val_indices

    def fold_count(self) -> int:
        return len(self.fold_splits)

    def build_holdout_trainval_loader(self, shuffle: bool = True):
        if not self.trainval_indices:
            raise RuntimeError("No trainval split is available. Set use_kfold=True.")
        trainval_dataset = self._build_split_dataset(self.trainval_indices, augment=self.augment_train_split)
        return self._build_loader(trainval_dataset, shuffle=shuffle)


    def save_model(self, model: nn.Module , path: str | None = None, save_optimizer: bool = False):
        """Save model state (and optional optimizer state) plus class list and training metrics."""
        path = path or self.save_path
        data = {
            "model_state_dict": model.state_dict(),
            "classes": self.classes,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "test_accuracy": self.test_accuracy,
            "epoch": len(self.train_accuracies)
        }
        if save_optimizer:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                data["optimizer_state_dict"] = self.optimizer.state_dict()
            if hasattr(self, "scheduler") and self.scheduler is not None:
                data["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(data, path)
        print(f"Saved model and metrics to: {path}")

    def load_model(self, model: nn.Module, path):
        checkpoint = torch.load(path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

            if (
                hasattr(self, "optimizer")
                and self.optimizer is not None
                and "optimizer_state_dict" in checkpoint
            ):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if (
                hasattr(self, "scheduler")
                and self.scheduler is not None
                and "scheduler_state_dict" in checkpoint
            ):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.train_losses = checkpoint.get("train_losses", [])
            self.val_losses = checkpoint.get("val_losses", [])
            self.train_accuracies = checkpoint.get("train_accuracies", [])
            self.val_accuracies = checkpoint.get("val_accuracies", [])
            self.test_accuracy = checkpoint.get("test_accuracy", None)
            self.start_epoch = checkpoint.get("epoch", 0)

            print(f"Model loaded from {path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Model loaded from {path} (legacy format)")

    def check_only_see_metrics(self, only_see_metrics: bool):
        if only_see_metrics:
            self.plot_metrics()
            exit()

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        train_acc_epochs = range(1, len(self.train_accuracies) + 1)
        val_acc_epochs = range(1, len(self.val_accuracies) + 1)
        train_loss_epochs = range(1, len(self.train_losses) + 1)
        val_loss_epochs = range(1, len(self.val_losses) + 1)

        ax1.plot(train_acc_epochs, self.train_accuracies, label='Train Accuracy', marker='o')
        if self.val_accuracies:
            ax1.plot(val_acc_epochs, self.val_accuracies, label='Validation Accuracy', marker='o')
        if self.test_accuracy is not None:
            test_epoch = (len(self.train_accuracies) + 1) if self.train_accuracies else 1
            ax1.scatter(
                [test_epoch],
                [self.test_accuracy],
                label=f'Test Accuracy ({self.test_accuracy:.2f}%)',
                marker='*',
                s=160,
                color='red',
                zorder=5,
            )
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training and/or Validation Accuracy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(train_loss_epochs, self.train_losses, label='Training Loss', marker='o')
        if self.val_losses:
            ax2.plot(val_loss_epochs, self.val_losses, label='Validation Loss', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()