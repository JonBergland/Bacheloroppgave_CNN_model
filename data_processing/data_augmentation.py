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

    def _compose_pipeline(self, transforms: list | None = None, preprocessed_input: bool = True):
        transforms = transforms or []
        ops = [] if preprocessed_input else self._preprocessed_ops()
        return v2.Compose(ops + transforms + self._end_ops())

    def getBaseTransform(self, preprocessed_input: bool = True):
        if preprocessed_input:
            return v2.Compose(self._end_ops())
        return v2.Compose(self._preprocessed_ops() + self._end_ops())

    def getDataAugmentations(
        self,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        translation: bool = False,
        blur: bool = False,
        color_jitter: bool = False,
        random_erasing: bool = False,
        preprocessed_input: bool = True,
    ):
        transforms = []

        if horizontal_flip:
            transforms.append(self._compose_pipeline([v2.RandomHorizontalFlip(p=1.0)], preprocessed_input=preprocessed_input))

        if vertical_flip:
            transforms.append(self._compose_pipeline([v2.RandomVerticalFlip(p=1.0)], preprocessed_input=preprocessed_input))

        if translation:
            transforms.append(self._compose_pipeline([v2.RandomAffine(degrees=0, translate=(0.1, 0.1))], preprocessed_input=preprocessed_input))

        if blur:
            transforms.append(self._compose_pipeline([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], preprocessed_input=preprocessed_input))

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
                    ],
                    preprocessed_input=preprocessed_input,
                )
            )

        if random_erasing:
            erasing_ops = [] if preprocessed_input else self._preprocessed_ops()
            erasing_ops.extend([
                self.toTensor(),
                v2.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                v2.Normalize(mean=[0.449], std=[0.226]),
            ])
            transforms.append(v2.Compose(erasing_ops))
        return transforms
