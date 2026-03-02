import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast

from resnet import ResNet18, ResNet9

class CNN():
    def __init__(self, 
            dataset_root: str, 
            epochs: int = 5,
            lr_rate: float = 0.01,
            momentum: float = 0.09,
            batch_size: int = 32,
            img_size: int = 32, 
            manual_seed: int = 42,
            save_path: str | None = None,
            only_see_metrics: bool = False):

        self.epochs = epochs
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226])
        ])

        self.batch_size = batch_size

        self.generator = torch.Generator().manual_seed(manual_seed)

        self.dataset = torchvision.datasets.ImageFolder(root=dataset_root, transform=transform)
        # trainset, valset, testset = torch.utils.data.random_split(dataset, [0.70, 0.15, 0.15], generator=generator)

        n = len(self.dataset)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        test_size = n - train_size - val_size

        trainset, valset, testset = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=self.generator
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        self.valloader = torch.utils.data.DataLoader(
            valset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        self.testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        self.classes = self.dataset.classes

        print(self.dataset.classes)

        # self.net = Net(num_classes=len(self.dataset.classes), img_size=img_size)

        # self.net = ResNet18(num_classes=len(self.dataset.classes), in_channels=1)
        self.net = ResNet9(num_classes=len(self.dataset.classes), in_channels=1)

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr_rate, amsgrad=True, weight_decay=0.01)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        # Initialize tracking lists
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.start_epoch = 0
        
        # Model save path
        if save_path is None:
            self.save_path = os.path.join(os.getcwd(), "resnet9_2.pth")
        else:
            print("Save path is not null")
            # If user passed a directory (or a path ending with separator), use default filename inside it
            if os.path.isdir(save_path) or str(save_path).endswith(os.sep):
                os.makedirs(save_path, exist_ok=True)
                self.save_path = os.path.join(save_path, "resnet9_2.pth")
                
            else:
                parent = os.path.dirname(save_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self.save_path = save_path

        # Auto-load existing checkpoint if present
        if os.path.exists(self.save_path):
            print("Save path exists from before")
            try:
                self.load_model(self.save_path, load_optimizer=True)
            except Exception as e:
                print(f"Warning: failed to load model from {self.save_path}: {e}")

        if only_see_metrics:
            self.plot_metrics()
            exit()


    def train(self):
        use_amp = (self.device_type == "cuda")
        scaler = GradScaler() if use_amp else None
        best_val_acc = max(self.val_accuracies) if self.val_accuracies else 0.0
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in self.trainloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                if use_amp:
                    with autocast(device_type=self.device_type):
                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            self.scheduler.step()
            avg_loss = running_loss / len(self.trainloader)
            train_acc = 100 * correct_train / total_train
            val_acc = self.validate()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(save_optimizer=True)
                print(f"New best model saved with Validation Accuracy: {val_acc:.2f}%")

            # Store metrics
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Val Acc: {val_acc:.2f}%"
            )
        


    def validate(self):
        self.net.eval()

        correct = 0
        total = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.valloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.net.train()

        avg_val_loss = running_val_loss / len(self.valloader)
        self.val_losses.append(avg_val_loss)
        
        accuracy = 100 * correct / total
        return accuracy


    def plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = range(1, len(self.train_accuracies) + 1)
        
        # Plot Accuracy
        ax1.plot(epochs_range, self.train_accuracies, label='Train Accuracy', marker='o')
        ax1.plot(epochs_range, self.val_accuracies, label='Validation Accuracy', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training vs Validation Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Loss
        ax2.plot(epochs_range, self.train_losses, label='Training Loss', marker='o')
        ax2.plot(epochs_range, self.val_losses, label='Validation Loss', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training vs Validation Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def clear_model(self):
        torch.cuda.empty_cache()


    def evaluate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)


                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Test image accuracy: %d %%' % (
            100 * correct / total))
        

    def save_model(self, path: str | None = None, save_optimizer: bool = False):
        """Save model state (and optional optimizer state) plus class list and training metrics."""
        path = path or self.save_path
        data = {
            "model_state_dict": self.net.state_dict(),
            "classes": self.classes,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "epoch": len(self.train_accuracies)
        }
        if save_optimizer:
            data["optimizer_state_dict"] = self.optimizer.state_dict()
            data["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(data, path)
        print(f"Saved model and metrics to: {path}")

    def load_model(self, path: str, load_optimizer: bool = False):
        """Load model (and optional optimizer/scheduler) state plus training metrics."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.classes = checkpoint.get("classes", self.classes)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.train_accuracies = checkpoint.get("train_accuracies", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])
        self.start_epoch = checkpoint.get("epoch", 0)
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.net.to(self.device)
        print(f"Loaded model from: {path}, resuming from epoch {self.start_epoch}")

