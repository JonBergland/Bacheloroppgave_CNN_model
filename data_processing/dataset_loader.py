from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from data_processing.data_augmentation import DataAugmentation


class DatasetLoader:
    """
    Loads the base dataset from the given root and can expand it with augmented copies.
    """

    def __init__(self, dataset_root: str, img_size: int, output_channels: int = 1):
        self.dataset_root = dataset_root
        self.img_size = img_size
        self.output_channels = output_channels

        self.data_augmentation = DataAugmentation(
            img_size=self.img_size,
            output_channels=self.output_channels,
        )

        self.base_dataset = ImageFolder(root=self.dataset_root, transform=self.data_augmentation.base_transforms)
        
    def getDataset(
        self,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        resizing: bool = False,
        translation: bool = False,
        blur: bool = False,
        color_jitter: bool = False,
        random_erasing: bool = False,
    ) -> Dataset:
        datasets: list[Dataset] = [self.base_dataset]

        augmentation_transforms = self.data_augmentation.getDataAugmentations(
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            resizing=resizing,
            translation=translation,
            blur=blur,
            color_jitter=color_jitter,
            random_erasing=random_erasing,
        )

        for transform in augmentation_transforms:
            datasets.append(ImageFolder(root=self.dataset_root, transform=transform))

        if len(datasets) == 1:
            return datasets[0]

        return ConcatDataset(datasets)

