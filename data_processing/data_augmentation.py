import torch
from torchvision.transforms import v2

class DataAugmentation:
    """
    A data class that deals with creating objects of torchvision's transforms-class
    """

    def __init__(self, img_size: int, output_channels: int = 1):
        self.img_size = img_size
        self.output_channels = output_channels

    def _preprocessed_ops(self):
        return [
            v2.Resize((self.img_size, self.img_size)),
            v2.Grayscale(num_output_channels=self.output_channels),
        ]

    def _end_ops(self):
        return [
            self.toTensor(),
            v2.Normalize(mean=[0.449], std=[0.226])
        ]

    def toTensor(self):
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def _compose_pipeline(self, transforms: list | None = None):
        transforms = transforms or []
        return v2.Compose(transforms + self._end_ops())

    def getBaseTransform(self):
        return self._compose_pipeline()

    def getDataAugmentations(
        self,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        translation: bool = False,
        blur: bool = False,
        color_jitter: bool = False,
        random_erasing: bool = False,
    ):
        transforms = []

        if horizontal_flip:
            transforms.append(self._compose_pipeline([v2.RandomHorizontalFlip(p=1.0)]))

        if vertical_flip:
            transforms.append(self._compose_pipeline([v2.RandomVerticalFlip(p=1.0)]))

        if translation:
            transforms.append(self._compose_pipeline([v2.RandomAffine(degrees=0, translate=(0.1, 0.1))]))

        if blur:
            transforms.append(self._compose_pipeline([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]))

        if color_jitter:
            transforms.append(
                self._compose_pipeline(
                    [
                        v2.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.0,
                            hue=0.0,
                        )
                    ]
                )
            )

        if random_erasing:
            transforms.append(self._compose_pipeline([v2.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3))]))

        return transforms

