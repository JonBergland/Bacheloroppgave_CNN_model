import torch
from torchvision.transforms import v2

class DataAugmentation:
    """
    A data class that deals with creating objects of torchvision's transforms-class
    """

    def __init__(self, img_size: int, output_channels: int = 1):
        self.img_size = img_size
        self.output_channels = output_channels

    def _not_preprocessed_ops(self):
        # Base Transforms for when the dataset is not preprocessed
        return [
            v2.Resize((self.img_size, self.img_size)),
            v2.Grayscale(num_output_channels=self.output_channels),
        ]

    def _preprocessed_ops(self):
        # Base Transforms for when the dataset is preprocessed 
        return [v2.Grayscale(num_output_channels=self.output_channels)]

    def _end_ops(self):
        return [
            self.toTensor(),
            v2.Normalize(mean=[0.449], std=[0.226])
        ]

    def toTensor(self):
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def _compose_pipeline(self, transforms: list | None = None, preprocessed_input: bool = True):
        transforms = transforms or []
        ops = self._preprocessed_ops() if preprocessed_input else self._not_preprocessed_ops()
        return v2.Compose(ops + transforms + self._end_ops())

    def _build_named_augmentations(
        self,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        translation: bool = False,
        blur: bool = False,
        random_erasing: bool = False,
        preprocessed_input: bool = True,
        noise_light: bool = False,
        noise_medium: bool = False,
        noise_strong: bool = False,
        scale: bool = False,
        exclude_base: bool = False,
    ):
        named_transforms = []

        if horizontal_flip:
            named_transforms.append(
                (
                    "Horizontal flip",
                    self._compose_pipeline([v2.RandomHorizontalFlip(p=1.0)], preprocessed_input=preprocessed_input),
                )
            )

        if vertical_flip:
            named_transforms.append(
                (
                    "vertical_flip",
                    self._compose_pipeline([v2.RandomVerticalFlip(p=1.0)], preprocessed_input=preprocessed_input),
                )
            )

        if translation:
            named_transforms.append(
                (
                    "Translation",
                    self._compose_pipeline(
                        [v2.RandomAffine(degrees=0, translate=(0.1, 0.1))],
                        preprocessed_input=preprocessed_input,
                    ),
                )
            )

        if blur:
            named_transforms.append(
                (
                    "blur",
                    self._compose_pipeline([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], preprocessed_input=preprocessed_input),
                )
            )

        if random_erasing:
            erasing_ops = self._preprocessed_ops() if preprocessed_input else self._not_preprocessed_ops()
            erasing_ops.extend([
                self.toTensor(),
                v2.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                v2.Normalize(mean=[0.449], std=[0.226]),
            ])
            named_transforms.append(("random_erasing", v2.Compose(erasing_ops)))

        if scale:
            zoom_crop_size = max(1, int(self.img_size * 0.85))
            named_transforms.append(
                (
                    "Scale Center Zoom",
                    self._compose_pipeline(
                        [
                            v2.CenterCrop(zoom_crop_size),
                            v2.Resize((self.img_size, self.img_size)),
                        ],
                        preprocessed_input=preprocessed_input,
                    ),
                )
            )

        if noise_light:
            noise_ops = self._preprocessed_ops() if preprocessed_input else self._not_preprocessed_ops()
            noise_ops.extend([
                self.toTensor(),
                v2.GaussianNoise(mean=0.0, sigma=0.05, clip=True),
                v2.Normalize(mean=[0.449], std=[0.226]),
            ])
            named_transforms.append(("Noise SD=0.05", v2.Compose(noise_ops)))

        if noise_medium:
            noise_ops = self._preprocessed_ops() if preprocessed_input else self._not_preprocessed_ops()
            noise_ops.extend([
                self.toTensor(),
                v2.GaussianNoise(mean=0.0, sigma=0.10, clip=True),
                v2.Normalize(mean=[0.449], std=[0.226]),
            ])
            named_transforms.append(("Noise SD=0.10", v2.Compose(noise_ops)))

        if noise_strong:
            noise_ops = self._preprocessed_ops() if preprocessed_input else self._not_preprocessed_ops()
            noise_ops.extend([
                self.toTensor(),
                v2.GaussianNoise(mean=0.0, sigma=0.15, clip=True),
                v2.Normalize(mean=[0.449], std=[0.226]),
            ])
            named_transforms.append(("Noise SD=0.15", v2.Compose(noise_ops)))

        return named_transforms

    def _denormalize_for_display(self, tensor: torch.Tensor):
        x = tensor.detach().cpu().clone()
        if x.dim() == 2:
            x = x.unsqueeze(0)

        channels = x.shape[0]
        mean = torch.full((channels, 1, 1), 0.449)
        std = torch.full((channels, 1, 1), 0.226)
        x = (x * std) + mean
        return x.clamp(0.0, 1.0)

    def getBaseTransform(self, preprocessed_input: bool = True):
        if preprocessed_input:
            return v2.Compose(self._preprocessed_ops() + self._end_ops())
        return v2.Compose(self._not_preprocessed_ops() + self._end_ops())

    def getDataAugmentations(
        self,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        translation: bool = False,
        blur: bool = False,
        random_erasing: bool = False,
        preprocessed_input: bool = True,
        noise_light: bool = False,
        noise_medium: bool = False,
        noise_strong: bool = False,
        scale: bool = False,
        exclude_base: bool = False
    ):
        named_transforms = self._build_named_augmentations(
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            translation=translation,
            blur=blur,
            random_erasing=random_erasing,
            preprocessed_input=preprocessed_input,
            noise_light=noise_light,
            noise_medium=noise_medium,
            noise_strong=noise_strong,
            scale=scale,
            exclude_base=exclude_base,
        )
        return [transform for _, transform in named_transforms]

    def showAugmentedSamples(
        self,
        image_path: str,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        translation: bool = True,
        blur: bool = True,
        random_erasing: bool = True,
        preprocessed_input: bool = True,
        noise_light: bool = True,
        noise_medium: bool = False,
        noise_strong: bool = False,
        scale: bool = True,
        include_base: bool = True,
    ):
        import matplotlib.pyplot as plt
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        samples = []

        if include_base:
            base_tensor = self.getBaseTransform(preprocessed_input=preprocessed_input)(image)
            samples.append(("Base", base_tensor))

        named_transforms = self._build_named_augmentations(
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            translation=translation,
            blur=blur,
            random_erasing=random_erasing,
            preprocessed_input=preprocessed_input,
            noise_light=noise_light,
            noise_medium=noise_medium,
            noise_strong=noise_strong,
            scale=scale,
        )

        for name, transform in named_transforms:
            samples.append((name, transform(image)))

        if not samples:
            print("No transforms were selected for preview.")
            return

        n = len(samples)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (name, tensor_image) in zip(axes, samples):
            image_to_show = self._denormalize_for_display(tensor_image)
            if image_to_show.shape[0] == 1:
                ax.imshow(image_to_show.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(image_to_show.permute(1, 2, 0))
            ax.set_title(name)
            ax.axis("off")

        plt.tight_layout()
        plt.show()
