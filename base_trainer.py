import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

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
                 output_channels: int = 1):

        self.epochs = epochs
        self.batch_size = batch_size
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # transforms.RandomHorizontalFlip(), # Data augmentations
            # transforms.RandomRotation(10),     # Data augmentation
            # Scale
            # Flytting
            transforms.Grayscale(num_output_channels=output_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226])
        ])

        self.generator = torch.Generator().manual_seed(manual_seed)

        self.dataset = torchvision.datasets.ImageFolder(root=dataset_root, transform=transform)

        n = len(self.dataset)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        test_size = n - train_size - val_size

        trainset, valset, testset = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=self.generator
        )

        use_pin_memory = self.device_type == "cuda"

        self.trainloader = DataLoader(
            trainset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=use_pin_memory,
            persistent_workers=True
        )

        self.valloader = DataLoader(
            valset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=use_pin_memory,
            persistent_workers=True
        )

        self.testloader = DataLoader(
            testset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=use_pin_memory,
            persistent_workers=True
        )

        self.classes = self.dataset.classes

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
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

        epochs_range = range(1, len(self.train_accuracies) + 1)

        ax1.plot(epochs_range, self.train_accuracies, label='Train Accuracy', marker='o')
        ax1.plot(epochs_range, self.val_accuracies, label='Validation Accuracy', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training vs Validation Accuracy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs_range, self.train_losses, label='Training Loss', marker='o')
        ax2.plot(epochs_range, self.val_losses, label='Validation Loss', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training vs Validation Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()