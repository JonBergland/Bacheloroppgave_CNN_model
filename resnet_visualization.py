from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch


NOISE_LEVELS = ["none", "light", "medium", "strong"]
SD_LEVELS = ["0.00", "0.05", "0.10", "0.15"]


@dataclass
class RunSummary:
	name: str
	train_aug: bool
	test_level: str
	noise_value: str
	test_accuracy_curve: list[float]


def _checkpoint_name(train_aug: bool, test_level: str) -> str:
	return f"resnet_trainAug_{'on' if train_aug else 'off'}_test_{test_level}.pth"


def _load_run_summary(checkpoint_path: Path, sd_value: str) -> RunSummary:
	checkpoint = torch.load(checkpoint_path, map_location="cpu")

	test_accuracy_curve = list(checkpoint.get("test_accuracies", []) or [])

	return RunSummary(
		name=checkpoint_path.stem,
		train_aug="trainAug_on" in checkpoint_path.stem,
		test_level=checkpoint_path.stem.rsplit("_test_", 1)[-1],
		noise_value=sd_value,
		test_accuracy_curve=test_accuracy_curve,
	)


def load_resnet_summaries(saved_models_dir: str | Path) -> list[RunSummary]:
	saved_models_path = Path(saved_models_dir)
	summaries: list[RunSummary] = []
	noise_to_sd = dict(zip(NOISE_LEVELS, SD_LEVELS))

	for train_aug in (False, True):
		for test_level in NOISE_LEVELS:
			checkpoint_path = saved_models_path / _checkpoint_name(train_aug, test_level)
			if not checkpoint_path.exists():
				print(f"Skipping missing checkpoint: {checkpoint_path.name}")
				continue
			summaries.append(_load_run_summary(checkpoint_path, noise_to_sd[test_level]))

	return summaries


def _plot_group(ax, summaries: list[RunSummary], title: str):
	colors = {
		"0.00": "#2b6cb0",
		"0.05": "#38a169",
		"0.10": "#d69e2e",
		"0.15": "#c53030",
	}

	for summary in sorted(summaries, key=lambda item: NOISE_LEVELS.index(item.test_level)):
		if not summary.test_accuracy_curve:
			continue
		epochs = list(range(1, len(summary.test_accuracy_curve) + 1))
		label = f"{summary.noise_value}"
		ax.plot(
			epochs,
			summary.test_accuracy_curve,
			label=label,
			color=colors.get(summary.noise_value, "#4a5568"),
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

	fig.suptitle("ResNet performance at different noise SD values", fontsize=14)
	fig.tight_layout()
	plt.show()


def print_summary_table(saved_models_dir: str | Path = "saved_models"):
	summaries = load_resnet_summaries(saved_models_dir)
	for summary in summaries:
		print(
			f"{summary.name}: train_aug={summary.train_aug}, test_level={summary.test_level}, "
			f"epochs={len(summary.test_accuracy_curve)}, final_test_accuracy={summary.test_accuracy_curve[-1] if summary.test_accuracy_curve else None}"
		)


def get_max_accuracy_table(saved_models_dir: str | Path = "saved_models") -> str:
	"""Generate a formatted table of maximum accuracy scores for all ResNet runs."""
	summaries = load_resnet_summaries(saved_models_dir)
	
	if not summaries:
		return "No summaries found"

	# Group by training augmentation
	off_summaries = sorted(
		[s for s in summaries if not s.train_aug],
		key=lambda x: NOISE_LEVELS.index(x.test_level)
	)
	on_summaries = sorted(
		[s for s in summaries if s.train_aug],
		key=lambda x: NOISE_LEVELS.index(x.test_level)
	)

	lines = []
	lines.append("ResNet Maximum Test Accuracy Scores")
	lines.append("=" * 60)
	lines.append("\nWithout Training Augmentation:")
	lines.append("-" * 60)
	lines.append(f"{'Noise Level':<15} {'Max Accuracy (%)':<20} {'Epoch':<10}")
	lines.append("-" * 60)

	for summary in off_summaries:
		if summary.test_accuracy_curve:
			max_acc = max(summary.test_accuracy_curve)
			max_epoch = summary.test_accuracy_curve.index(max_acc) + 1
			noise_label = summary.test_level.replace("none", "No Noise").title()
			lines.append(f"{noise_label:<15} {max_acc:<20.2f} {max_epoch:<10}")

	lines.append("\nWith Training Augmentation:")
	lines.append("-" * 60)
	lines.append(f"{'Noise Level':<15} {'Max Accuracy (%)':<20} {'Epoch':<10}")
	lines.append("-" * 60)

	for summary in on_summaries:
		if summary.test_accuracy_curve:
			max_acc = max(summary.test_accuracy_curve)
			max_epoch = summary.test_accuracy_curve.index(max_acc) + 1
			noise_label = summary.test_level.replace("none", "No Noise").title()
			lines.append(f"{noise_label:<15} {max_acc:<20.2f} {max_epoch:<10}")

	return "\n".join(lines)


def get_max_accuracy_csv(saved_models_dir: str | Path = "saved_models") -> str:
	"""Generate a CSV table of maximum accuracy scores for all ResNet runs."""
	summaries = load_resnet_summaries(saved_models_dir)
	
	if not summaries:
		return ""

	summaries = sorted(summaries, key=lambda x: (x.train_aug, NOISE_LEVELS.index(x.test_level)))

	lines = ["Model,Training Augmentation,Noise Level,Max Accuracy (%),Epoch"]
	
	for summary in summaries:
		if summary.test_accuracy_curve:
			max_acc = max(summary.test_accuracy_curve)
			max_epoch = summary.test_accuracy_curve.index(max_acc) + 1
			train_aug_str = "Yes" if summary.train_aug else "No"
			lines.append(f"ResNet,{train_aug_str},{summary.test_level},{max_acc:.2f},{max_epoch}")

	return "\n".join(lines)


if __name__ == "__main__":
	plot_resnet_augmentation_groups()
