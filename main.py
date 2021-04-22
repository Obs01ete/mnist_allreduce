from __future__ import print_function
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class DataparallelModel:
    def __init__(self, net_factory, optimizer_factory, loss_functor, num_replicas, reference_model=None):
        self.net_factory = net_factory
        self.optimizer_factory = optimizer_factory
        self.loss_functor = loss_functor
        self.num_replicas = num_replicas
        models = []
        optimizers = []
        for i_replica in range(num_replicas):
            model = net_factory()
            models.append(model)
            optimizer = self.optimizer_factory(model.parameters())
            optimizers.append(optimizer)
        self.models = models
        self.optimizers = optimizers

        if reference_model is not None:
            for param_group, ref_param in zip(self.param_group_gen(), reference_model.named_parameters()):
                for param in param_group:
                    param.data[...] = ref_param[1].data[...]
        else:
            for param_group in self.param_group_gen():
                for param in param_group[1:]:
                    param.data[...] = param_group[0].data[...]

        self.current_loss = None

    def param_group_gen(self):
        param_groups = [m.parameters() for m in self.models]
        for group in zip(*param_groups):
            yield group

    def step(self, data, target):

        losses = []
        for i_replica, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            optimizer.zero_grad()
            output = model(data)
            loss = self.loss_functor(output, target)
            loss.backward()
            losses.append(loss.item())

        self.current_loss = np.mean(np.array(losses))

        # gradient allreduce here

        for param_group in self.param_group_gen():
            param_group_data = tuple(p.data for p in param_group)
            sum_tensor = torch.sum(torch.stack(param_group_data, dim=0), dim=0)
            for param in param_group:
                param.data[...] = sum_tensor[...]

        for i_replica, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            optimizer.step()

        # check all replica weights are identical

        return

    def get_loss(self):
        return 0.0

    def named_gradients(self):
        assert len(self.models) > 0
        def grad_gen():
            for param in self.models[0].named_parameters():
                yield param[0], param[1].data.grad
        return grad_gen()

    def named_parameters(self):
        assert len(self.models) > 0
        return self.models[0].named_parameters()


class Trainer:
    def __init__(self, args):
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset1 = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)
        self.dataset2 = datasets.MNIST('../data', train=False,
                                  transform=transform)
        self.train_loader = torch.utils.data.DataLoader(self.dataset1, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.dataset2, **test_kwargs)

        def net_factory():
            return Net()

        def optimizer_factory(params):
            return optim.Adadelta(params, lr=args.lr)

        self.model = net_factory()
        self.optimizer = optimizer_factory(self.model.parameters())

        num_replicas = 4
        self.dataparallel_model = DataparallelModel(net_factory, optimizer_factory, F.nll_loss, num_replicas,
                                                    reference_model=self.model)

        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args.gamma)

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            self.test()
            self.scheduler.step()
            if self.args.dry_run:
                break

        if self.args.save_model:
            torch.save(self.model.state_dict(), "mnist_cnn.pt")

    def train_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.dataparallel_model.step(data, target)
            dp_loss = self.dataparallel_model.get_loss()
            # dp_gradients = self.dataparallel_model.named_gradients()
            dp_parameters = self.dataparallel_model.named_parameters()

            print(f"Loss, reference={loss.item()} dp={dp_loss}")

            # for ref_np, dp_np in zip(self.model.named_parameters(), dp_parameters):
            #     print(ref_np, dp_np)

            # def grad_gen():
            #     for param in self.models[0].parameters():
            #         yield param.data.grad

            for ref_np, dp_np in zip(self.model.named_parameters(), dp_parameters):
                print(ref_np[0], dp_np[0])
                print(ref_np[1].grad[0, 0, ...])
                print(dp_np[1].grad[0, 0, ...])
                print("")

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
