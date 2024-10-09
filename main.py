import argparse
import torch
from model import ResNet18
from data.utils import CIFAR, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet model on CIFAR10")

    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--pretrained", type=str, default='./checkpoint/best_model.pth', help="Path to pretrained weights (.pth)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cifar_data = CIFAR(batch_size=args.batch_size, num_workers=args.num_workers)

    model = ResNet18()

    print(f"Loading pretrained weights from {args.pretrained}...")
    try:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint)
    except:
        print('Loading pretrianed weights failed')

    best_acc = train_model(
        model, cifar_data, lr=args.learning_rate, epochs=args.epochs, device=device
    )

    print(f"Training complete. Best accuracy achieved: {best_acc:.3f}%")
