import numpy as np
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


        # TODO create an original dataset that is the dataset without any augmentations. 
        # Create subsets from there.
        base_dataset = ImageFolder(root=self.dataset_root, transform=self.data_augmentation.getBaseTransform())
        self.classes = base_dataset.classes
        self.datasets: list[Dataset] = [base_dataset]


    def getDataAugmentations(
            self,
            use_augmentations: bool = False
        ):
        if use_augmentations:
            return self.data_augmentation.getDataAugmentations(
                horizontal_flip=True,
                vertical_flip=True,
                resizing=True,
                translation=True,
                blur=True,
                color_jitter=True,
                random_erasing=True,
            )
        else:
            return self.data_augmentation.getDataAugmentations()
        
    def getDataset(
        self,
        use_augmentations: bool = False,
        test_ratio: float = 0.15
        ) -> Dataset:
        
        ## PSUDO KODE
        ## TRAIN - TEST split


        augmentation_transforms = self.getDataAugmentations(use_augmentations=use_augmentations)

        for transform in augmentation_transforms:
            self.datasets.append(ImageFolder(root=self.dataset_root, transform=transform))

        return ConcatDataset(self.datasets)
    
    def getDatasetClasses(self):
        return self.classes
    
    def getDatasetTargets(self): 
        all_targets = []
        for ds in self.dataset.datasets:
            all_targets.extend(ds.targets) 
        return np.array(all_targets)


