import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from vision_transformer import VisionTransformer
from base_trainer import BaseTrainer

class VisionTransformerTrainer(BaseTrainer):
    def __init__(self, 
                 dataset_root: str, 
                 model_name: str,
                 epochs: int = 5,
                 lr_rate: float = 0.01,
                 batch_size: int = 32,
                 img_size: int = 64, 
                 manual_seed: int = 42,
                 save_path: str | None = None,
                 only_see_metrics: bool = False,
                 dropout_rate: float = 0.6,
                 label_smoothing: float = 0.05,
                 weight_decay: float = 2e-3,
                 lr_step_size: int = 5,
                 lr_gamma: float = 0.5):
        super().__init__(dataset_root, model_name, epochs, lr_rate, batch_size, img_size, manual_seed, save_path)

        self.model = VisionTransformer(
            img_size=img_size,
            patch_size=8,
            num_classes=len(self.classes),
            embed_dim=288,
            num_heads=6,
            depth=6,
            mlp_dim=864,
            in_channels=1,
            dropout=dropout_rate
        )
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr_rate, amsgrad=True, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_step_size, gamma=lr_gamma
        )

        if os.path.exists(self.save_path):
            try:
                self.load_model(self.model, self.save_path)
            except Exception as e:
                print(f"Warning: failed to load model from {self.save_path}: {e}")

        self.check_only_see_metrics(only_see_metrics)

    def train(self):
        best_val_acc = max(self.val_accuracies) if self.val_accuracies else 0.0
        print("Starting to train")

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
                self.save_model(model=self.model, save_optimizer=True)
                print(f"New best model saved with Validation Accuracy: {val_acc:.2f}%")

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(
                f"Epoch [{epoch+1}/{self.start_epoch + self.epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Val Acc: {val_acc:.2f}%"
            )

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = running_val_loss / len(self.valloader)
        self.val_losses.append(avg_val_loss)

        accuracy = 100 * correct / total
        return accuracy
    
    def evaluate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)


                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Test image accuracy: %d %%' % (
            100 * correct / total))
        
    def clear_model(self):
        self.model = None
        if self.device_type == "cuda":
            torch.cuda.empty_cache()