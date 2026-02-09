import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from net import Net

class CNN():
    def __init__(self, 
            dataset_root: str, 
            epochs: int = 5,
            lr_rate: float = 0.01,
            momentum: float = 0.09,
            batch_size: int = 32,
            img_size: int = 32, 
            manual_seed: int = 42):

        self.epochs = epochs
        
        # Setting transforms to resizing the images
        transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.449], std=[0.226])])

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
            num_workers=4,
            pin_memory=True
        )

        self.valloader = torch.utils.data.DataLoader(
            valset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )

        self.testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )

        self.classes = self.dataset.classes

        print(self.dataset.classes)

        self.net = Net(num_classes=len(self.dataset.classes), img_size=img_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr_rate)


        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )


    def train(self):
        for epoch in range(self.epochs):

            running_loss = 0.0

            for inputs, labels in self.trainloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.trainloader)
            val_acc = self.validate()

            print(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Val Acc: {val_acc:.2f}%"
            )


    def validate(self):
        self.net.eval()  # evaluation mode

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.valloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.net.train()  # back to training mode

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


                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Test image accuracy: %d %%' % (
            100 * correct / total))


