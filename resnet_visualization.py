from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch


NOISE_LEVELS = ["none", "light", "medium", "strong"]


@dataclass
class RunSummary:
	name: str
	train_aug: bool
	test_level: str
	test_accuracy_curve: list[float]


def _checkpoint_name(train_aug: bool, test_level: str) -> str:
	return f"resnet_trainAug_{'on' if train_aug else 'off'}_test_{test_level}.pth"


def _load_run_summary(checkpoint_path: Path) -> RunSummary:
	checkpoint = torch.load(checkpoint_path, map_location="cpu")

	test_accuracy_curve = list(checkpoint.get("test_accuracies", []) or [])

	return RunSummary(
		name=checkpoint_path.stem,
		train_aug="trainAug_on" in checkpoint_path.stem,
		test_level=checkpoint_path.stem.rsplit("_test_", 1)[-1],
		test_accuracy_curve=test_accuracy_curve,
	)


def load_resnet_summaries(saved_models_dir: str | Path) -> list[RunSummary]:
	saved_models_path = Path(saved_models_dir)
	summaries: list[RunSummary] = []

	for train_aug in (False, True):
		for test_level in NOISE_LEVELS:
			checkpoint_path = saved_models_path / _checkpoint_name(train_aug, test_level)
			if not checkpoint_path.exists():
				print(f"Skipping missing checkpoint: {checkpoint_path.name}")
				continue
			summaries.append(_load_run_summary(checkpoint_path))

	return summaries


def _plot_group(ax, summaries: list[RunSummary], title: str):
	colors = {
		"none": "#2b6cb0",
		"light": "#38a169",
		"medium": "#d69e2e",
		"strong": "#c53030",
	}

	for summary in sorted(summaries, key=lambda item: NOISE_LEVELS.index(item.test_level)):
		if not summary.test_accuracy_curve:
			continue
		epochs = list(range(1, len(summary.test_accuracy_curve) + 1))
		label = summary.test_level.replace("none", "no noise").title()
		ax.plot(
			epochs,
			summary.test_accuracy_curve,
			label=label,
			color=colors.get(summary.test_level, "#4a5568"),
			linewidth=2,
		)

	ax.set_title(title)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Test accuracy (%)")
	ax.set_xlim(left=1)
	ax.grid(True, alpha=0.25)
	ax.legend(loc="lower right")


def plot_resnet_augmentation_groups(saved_models_dir: str | Path = "saved_models"):
	summaries = load_resnet_summaries(saved_models_dir)

	if not summaries:
		raise FileNotFoundError(f"No ResNet checkpoints found in {saved_models_dir}")

	off_summaries = [summary for summary in summaries if not summary.train_aug]
	on_summaries = [summary for summary in summaries if summary.train_aug]

	fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
	_plot_group(axes[0], sorted(off_summaries, key=lambda item: NOISE_LEVELS.index(item.test_level)), "ResNet without training augmentation")
	_plot_group(axes[1], sorted(on_summaries, key=lambda item: NOISE_LEVELS.index(item.test_level)), "ResNet with training augmentation")

	fig.suptitle("ResNet comparison by training augmentation setting", fontsize=14)
	fig.tight_layout()
	plt.show()


def print_summary_table(saved_models_dir: str | Path = "saved_models"):
	summaries = load_resnet_summaries(saved_models_dir)
	for summary in summaries:
		print(
			f"{summary.name}: train_aug={summary.train_aug}, test_level={summary.test_level}, "
			f"epochs={len(summary.test_accuracy_curve)}, final_test_accuracy={summary.test_accuracy_curve[-1] if summary.test_accuracy_curve else None}"
		)


if __name__ == "__main__":
	plot_resnet_augmentation_groups()
