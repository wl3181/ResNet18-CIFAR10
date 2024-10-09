import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm

class CIFAR:
    """
    Initialize CIFAR dataset (CIFAR-10 in this case).
    This class prepares the train and test data loaders.
    """

    def __init__(self, batch_size=128, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader, self.test_loader = self.prepare_data()

    def prepare_data(self):
        """
        Prepare train and test data loaders with transforms.
        """
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=self.num_workers
        )

        return trainloader, testloader


def train_model(model, cifar_data, lr=0.1, epochs=200, device="cuda"):
    """
    Function to train and evaluate the model on CIFAR dataset.

    Args:
        model: The neural network model to be trained.
        cifar_data: An object of the CIFAR class that holds train and test loaders.
        lr: Learning rate.
        epochs: Number of training epochs.
        device: Device to use ('cuda' or 'cpu').

    Returns:
        best_acc: Best accuracy achieved on the test dataset.
    """
    # Move model to the specified device (GPU or CPU)
    model = model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        with tqdm(total=len(cifar_data.train_loader), desc=f"Train Epoch {epoch+1}") as pbar:
            for batch_idx, (inputs, targets) in enumerate(cifar_data.train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({"Loss": f"{train_loss/(batch_idx+1):.3f}", 
                                  "Acc": f"{100.0 * correct / total:.3f}%"})
                pbar.update(1)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(cifar_data.test_loader), desc=f"Test Epoch {epoch+1}") as pbar:
                for batch_idx, (inputs, targets) in enumerate(cifar_data.test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    pbar.set_postfix({"Loss": f"{test_loss/(batch_idx+1):.3f}", 
                                      "Acc": f"{100.0 * correct / total:.3f}%"})
                    pbar.update(1)

        acc = 100.0 * correct / total
        print(f"Test Accuracy: {acc:.3f}%")

        if acc > best_acc:
            print(f"Saving best model with accuracy: {acc:.3f}%")
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(model.state_dict(), "./checkpoint/best_model.pth")
            best_acc = acc

        scheduler.step()

    return best_acc
