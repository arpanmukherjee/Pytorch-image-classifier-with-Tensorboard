import os
import torch
import warnings
import argparse
import torchvision
from model import Net
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_correct(y_pred, y_true):
    return y_pred.argmax(dim=1).eq(y_true).sum().item()


def train(args, model, train_loader, optimizer, criterion):
    model.train()

    img, lbl = iter(train_loader).next()
    img_grid = torchvision.utils.make_grid(img)

    tboard = SummaryWriter()
    tboard.add_image('images', img_grid)

    for n_epoch in range(args.epochs):
        batch_loss = 0.0
        total_loss = 0.0
        batch_correct = 0
        total_correct = 0
        for batch_idx, (X, y_true) in enumerate(train_loader):
            y_pred = model(X)

            loss = criterion(y_pred, y_true)  # Calculate loss
            optimizer.zero_grad()  # Zero parameter gradients
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

            # print loss
            batch_loss += loss.item()
            total_loss += loss.item()
            batch_correct += get_correct(y_pred, y_true)
            total_correct += get_correct(y_pred, y_true)
            if batch_idx % args.log_interval == 0:
                print('[%d, %5d] Total loss: %.5f, Total Correct: %5d' %
                      (n_epoch + 1, batch_idx + 1, batch_loss / args.log_interval, batch_correct))
                batch_loss = 0.0
                batch_correct = 0

        tboard.add_scalar('Loss', total_loss, n_epoch)
        tboard.add_scalar('Accuracy', total_correct /
                          len(train_loader.dataset), n_epoch)

        tboard.add_histogram('Conv1.bias', model.conv1.bias, n_epoch)
        tboard.add_histogram('Conv1.weight', model.conv1.weight, n_epoch)
        tboard.add_histogram('Conv1.weight.grad',
                             model.conv1.weight.grad, n_epoch)

    tboard.add_graph(model, img)
    tboard.close()
    print('Finished Training')
    if args.save_model:
        torch.save(model.state_dict(),
                   ('CNNModel_' + args.dataset + '.pth'))


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (X, y_true) in enumerate(test_loader):
            y_pred = model(X)
            total += y_true.size(0)
            correct += get_correct(y_pred, y_true)
        print('Accuracy of the network on test images: %d %%' % (
            100 * correct / total))


def main():
    parser = argparse.ArgumentParser(
        description='Simple training script for training model')

    parser.add_argument(
        '--epochs', help='Number of epochs (default: 75)', type=int, default=75)
    parser.add_argument(
        '--batch-size', help='Batch size of the data (default: 16)', type=int, default=16)
    parser.add_argument(
        '--learning-rate', help='Learning rate (default: 0.001)', type=float, default=0.001)
    parser.add_argument(
        '--seed', help='Random seed (default:1)', type=int, default=1)
    parser.add_argument(
        '--data-path', help='Path for the downloaded dataset (default: ../dataset/)', default='../dataset/')
    parser.add_argument(
        '--dataset', help='Dataset name. Must be one of MNIST, STL10, CIFAR10')
    parser.add_argument(
        '--use-cuda', help='CUDA usage (default: False)', type=bool, default=False)
    parser.add_argument(
        '--weight-decay', help='weight decay (L2 penalty) (default: 1e-5)', type=float, default=1e-5)
    parser.add_argument(
        '--log-interval', help='No of batches to wait before logging training status (default: 50)', type=int, default=50)
    parser.add_argument(
        '--save-model', help='For saving the current model (default: True)', type=bool, default=True)

    args = parser.parse_args()

    batch_size = args.batch_size  # batch size
    learning_rate = args.learning_rate  # learning rate
    torch.manual_seed(args.seed)  # seed value

    # Creating dataset path if it doesn't exist
    if args.data_path is None:
        raise ValueError('Must provide dataset path')
    else:
        data_path = args.data_path
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

    # Downloading proper dataset and creating data loader
    if args.dataset == 'MNIST':
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_data = torchvision.datasets.MNIST(
            data_path, train=True, download=True, transform=T)
        test_data = torchvision.datasets.MNIST(
            data_path, train=False, download=True, transform=T)
    elif args.dataset == 'STL10':
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = torchvision.datasets.STL10(
            data_path, split='train', download=True, transform=T)
        test_data = torchvision.datasets.STL10(
            data_path, split='test', download=True, transform=T)
    elif args.dataset == 'CIFAR10':
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = torchvision.datasets.CIFAR10(
            data_path, train=True, download=True, transform=T)
        test_data = torchvision.datasets.CIFAR10(
            data_path, train=False, download=True, transform=T)
    elif args.dataset is None:
        raise ValueError('Must provide dataset')
    else:
        raise ValueError('Dataset name must be MNIST, STL10 or CIFAR10')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # use CUDA or not
    device = 'cpu'
    if args.use_cuda is False:
        if torch.cuda.is_available():
            warnings.warn(
                'CUDA is available, please use for faster convergence')
        else:
            device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            raise ValueError('CUDA is not available, please set it False')

    # Create the model
    model = Net(dataset=args.dataset).to(device)

    # Train the network
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        lr=learning_rate, params=model.parameters(), weight_decay=args.weight_decay)
    train(args, model, device, train_loader, optimizer, criterion)

    # Test the network
    test(model, test_loader)


if __name__ == '__main__':
    main()
