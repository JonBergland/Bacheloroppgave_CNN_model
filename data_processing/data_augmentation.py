from torchvision.transforms import v2


class DataAugmentation:
    """
    A data class that deals with creating objects of torchvision's transforms-class
    """

    def __init__(self, img_size: int, output_channels: int = 1):
        self.img_size = img_size
        self.output_channels = output_channels

        self.base_transforms = [
            v2.Resize((self.img_size, self.img_size)),
            v2.Grayscale(num_output_channels=self.output_channels),
            v2.ToTensor(),
            v2.Normalize(mean=[0.449], std=[0.226]),
        ]

    def getBaseTransform(self):
        return self.base_transforms

    def getDataAugmentations(
        self,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        resizing: bool = False,
        translation: bool = False,
        blur: bool = False,
        color_jitter: bool = False,
        random_erasing: bool = False,
    ):
        transforms = []

        if horizontal_flip:
            t = self.base_transforms.copy()
            t.insert(1, v2.RandomHorizontalFlip(p=1.0))
            transforms.append(v2.Compose(t))

        if vertical_flip:
            t = self.base_transforms.copy()
            t.insert(1, v2.RandomVerticalFlip(p=1.0))
            transforms.append(v2.Compose(t))

        if resizing:
            t = self.base_transforms.copy()
            t[0] = v2.RandomResizedCrop(
                size=(self.img_size, self.img_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            )
            transforms.append(v2.Compose(t))

        if translation:
            t = self.base_transforms.copy()
            t.insert(1, v2.RandomAffine(degrees=0, translate=(0.1, 0.1)))
            transforms.append(v2.Compose(t))

        if blur:
            t = self.base_transforms.copy()
            t.insert(1, v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
            transforms.append(v2.Compose(t))

        if color_jitter:
            t = self.base_transforms.copy()
            t.insert(
                1,
                v2.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
            )
            transforms.append(v2.Compose(t))

        if random_erasing:
            t = self.base_transforms.copy()
            t.append(v2.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
            transforms.append(v2.Compose(t))

        return transforms

