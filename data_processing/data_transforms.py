"""Central transform dictionaries used by dataset split builders."""

# Per-split augmentation options used by DatasetSplitter.getDataAugmentations(...).
# You can override these from BaseTrainer if needed.
DEFAULT_SPLIT_TRANSFORMS = {
	"train": {
		"horizontal_flip": True,
		"vertical_flip": False,
		"translation": True,
		"blur": False,
		"random_erasing": False,
		"noise_light": False,
		"noise_medium": False,
		"noise_strong": False,
		"scale": True,
	},
	"val": {
		"horizontal_flip": False,
		"vertical_flip": False,
		"translation": False,
		"blur": False,
		"random_erasing": False,
		"noise_light": True,
		"noise_medium": False,
		"noise_strong": False,
		"scale": False,
	},
	"test": {
		"horizontal_flip": False,
		"vertical_flip": False,
		"translation": False,
		"blur": False,
		"random_erasing": False,
		"noise_light": True,
		"noise_medium": False,
		"noise_strong": False,
		"scale": False,
	},
}