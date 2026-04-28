import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from base_trainer import BaseTrainer

from resnet import ResNet9

class ResnetTrainer(BaseTrainer):
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
                 lr_gamma: float = 0.5,
                 use_kfold: bool = False,
                 n_splits: int = 5,
                 test_ratio: float = 0.15,
                 stratified_kfold: bool = True,
                 num_workers: int = 1,
                 augment_train_split: bool = False,
                 augment_test_split: bool = False,
                 dataset_is_preprocessed: bool = True,
                 depth: int = 6,
                 patience: int = 15,
                 use_val_split: bool = True):
        super().__init__(
            dataset_root=dataset_root,
            model_name=model_name,
            epochs=epochs,
            lr_rate=lr_rate,
            batch_size=batch_size,
            img_size=img_size,
            manual_seed=manual_seed,
            save_path=save_path,
            output_channels=1,
            use_kfold=use_kfold,
            n_splits=n_splits,
            test_ratio=test_ratio,
            stratified_kfold=stratified_kfold,
            num_workers=num_workers,
            augment_train_split=augment_train_split,
            augment_test_split=augment_test_split,
            dataset_is_preprocessed=dataset_is_preprocessed,
            use_val_split=use_val_split,
        )

        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.weight_decay = weight_decay
        self.lr_rate = lr_rate
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.depth = depth
        self.patience = patience
        self.patience_counter = 0

        self._initialize_model_components()

        if (not self.use_kfold) and os.path.exists(self.save_path):
            try:
                self.load_model(self.model, self.save_path)
            except Exception as e:
                print(f"Warning: failed to load model from {self.save_path}: {e}")

        if not self.use_kfold:
            self.check_only_see_metrics(only_see_metrics)

    def _initialize_model_components(self):
        self.model = ResNet9(num_classes=len(self.classes), in_channels=1, dropout_rate=self.dropout_rate)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr_rate,
            amsgrad=True,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma,
        )

    def reset_for_new_fold(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.test_losses_epoch = []
        self.start_epoch = 0
        self.patience_counter = 0
        self._initialize_model_components()

    def train(self):
        use_amp = (self.device_type == "cuda")
        scaler = GradScaler() if use_amp else None
        
        if self.use_val_split:
            best_acc = max(self.val_accuracies) if self.val_accuracies else 0.0
            eval_loader = self.valloader
            eval_name = "Validation"
        else:
            best_acc = max(self.test_accuracies) if hasattr(self, 'test_accuracies') and self.test_accuracies else 0.0
            if not hasattr(self, 'test_accuracies'):
                self.test_accuracies = []
            eval_loader = self.testloader
            eval_name = "Test"
        
        print("Starting to train")

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_idx, (inputs, labels) in enumerate(self.trainloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                if use_amp:
                    with autocast(device_type=self.device_type):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

                if batch_idx % 20 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.start_epoch + self.epochs} | "
                        f"Batch {batch_idx + 1}/{len(self.trainloader)} | "
                        f"Seen {total_train} samples"
                    )

            self.scheduler.step()
            avg_loss = running_loss / len(self.trainloader)
            train_acc = 100 * correct_train / total_train
            eval_acc = None

            if eval_loader is not None:
                if self.use_val_split:
                    eval_acc = self.validate()
                else:
                    eval_acc = self.evaluate_epoch()

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(train_acc)
            if eval_acc is not None:
                if self.use_val_split:
                    self.val_accuracies.append(eval_acc)
                else:
                    self.test_accuracies.append(eval_acc)

            if eval_acc is not None:
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    self.patience_counter = 0
                    self.save_model(model=self.model, save_optimizer=True)
                    print(f"New best model saved with {eval_name} Accuracy: {eval_acc:.2f}%")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs (patience={self.patience})")
                        break

            status = (
                f"Epoch [{epoch+1}/{self.start_epoch + self.epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Train Acc: {train_acc:.2f}%"
            )
            if eval_acc is not None:
                status += f" {eval_name} Acc: {eval_acc:.2f}% (patience: {self.patience_counter}/{self.patience})"
            print(status)

    def validate(self):
        self.model.eval()

        correct = 0
        total = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.valloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.model.train()

        avg_val_loss = running_val_loss / len(self.valloader)
        self.val_losses.append(avg_val_loss)

        accuracy = 100 * correct / total
        return accuracy

    def evaluate_epoch(self):
        """Evaluate model on test set for a single epoch during training. Returns accuracy without saving."""
        self.model.eval()

        correct = 0
        total = 0
        running_test_loss = 0.0

        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_test_loss += loss.item()

                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.model.train()

        avg_test_loss = running_test_loss / len(self.testloader)
        self.test_losses_epoch.append(avg_test_loss) if hasattr(self, 'test_losses_epoch') else None

        accuracy = 100 * correct / total
        return accuracy

    def restore_best_model(self):
        """Restore the best model from the saved checkpoint after early stopping."""
        if os.path.exists(self.save_path):
            self.load_model(self.model, self.save_path)
            print(f"Best model restored from {self.save_path}")
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        running_test_loss = 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)


                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.test_accuracy = 100 * correct / total
        self.test_loss = running_test_loss / len(self.testloader)
        print('Test image accuracy: %d %%' % (self.test_accuracy))


    def clear_model(self):
        self.model = None
        if self.device_type == "cuda":
            torch.cuda.empty_cache()