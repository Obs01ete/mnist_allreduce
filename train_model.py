import argparse
from functools import partial
import contextlib

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
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
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


class GenericModel:
    def __init__(self):
        self.is_train = True

    def train(self, is_train: bool = True):
        self.is_train = is_train


class ReferenceModel(GenericModel):
    def __init__(self, net_factory, optimizer_factory, lr_scheduler_factory, loss_functor):
        super().__init__()

        self.net_factory = net_factory
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.loss_functor = loss_functor

        self.model: nn.Module = net_factory()
        self.optimizer = optimizer_factory(self.model.parameters())
        self.scheduler = self.lr_scheduler_factory(self.optimizer)

    def step(self, data, target, no_grad=False, dry_run=False):
        if not no_grad:
            self.optimizer.zero_grad()
        with torch.no_grad() if no_grad else contextlib.nullcontext():
            output = self.model(data)
            loss = self.loss_functor(output, target)
        if not no_grad:
            loss.backward()
        if not dry_run and not no_grad:
            self.optimizer.step()
        return dict(loss=loss.item(), output=output.detach())

    def get_model(self):
        return self.model

    def named_parameters(self):
        return self.model.named_parameters()

    def train(self, is_train: bool = True):
        super().train(is_train)
        self.model.train(is_train)

    def lr_scheduler_step(self):
        self.scheduler.step()


class DataparallelModel(GenericModel):
    def __init__(self, net_factory, optimizer_factory, lr_scheduler_factory,
                 loss_functor, num_replicas, reference_model=None):
        super().__init__()

        self.net_factory = net_factory
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.loss_functor = loss_functor
        self.num_replicas = num_replicas

        models = []
        optimizers = []
        schedulers = []
        for i_replica in range(num_replicas):
            model = net_factory()
            models.append(model)
            optimizer = self.optimizer_factory(model.parameters())
            optimizers.append(optimizer)
            scheduler = self.lr_scheduler_factory(optimizer)
            schedulers.append(scheduler)

        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers

        self.master_model_idx = 0

        if reference_model is not None:
            # If a reference model is given, broadcast it
            for param_group, ref_param in zip(self.param_group_gen(), reference_model.named_parameters()):
                for param in param_group:
                    param.data[...] = ref_param[1].data[...]
        else:
            # If there is no reference model, broadcast weights of the master model
            for param_group in self.param_group_gen():
                assert self.master_model_idx == 0
                for param in param_group[1:]:
                    param.data[...] = param_group[self.master_model_idx].data[...]

    def param_group_gen(self):
        param_groups = [m.parameters() for m in self.models]
        for group in zip(*param_groups):
            yield group

    def step(self, data, target, no_grad=False, dry_run=False):

        assert len(data) % self.num_replicas == 0
        offset = len(data) // self.num_replicas

        losses = []
        outputs = []
        for i_replica, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            data_rep = data[i_replica*offset:(i_replica+1)*offset]
            target_rep = target[i_replica*offset:(i_replica+1)*offset]
            if not no_grad:
                optimizer.zero_grad()
            with torch.no_grad() if no_grad else contextlib.nullcontext():
                output = model(data_rep)
                loss = self.loss_functor(output, target_rep)
            if not no_grad:
                loss.backward()
            losses.append(loss.item())
            outputs.append(output.detach())

        total_loss = np.mean(np.array(losses))

        outputs = torch.cat(outputs, dim=0)

        if not no_grad:
            for param_group in self.param_group_gen():
                param_group_data = tuple(p.grad for p in param_group)

                # Modify gradient allreduce here
                # Below is a star-allreduce implementation. Replace it with your own.
                reduced_tensor = torch.mean(torch.stack(param_group_data, dim=0), dim=0)
                for grad in param_group_data:
                    grad[...] = reduced_tensor[...]

        if not dry_run and not no_grad:
            for i_replica, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                optimizer.step()

        # check all replica weights are identical

        return dict(loss=total_loss, pred=outputs)

    def named_gradients(self):
        assert len(self.models) > 0
        def grad_gen():
            for param in self.models[self.master_model_idx].named_parameters():
                yield param[0], param[1].data.grad
        return grad_gen()

    def named_parameters(self):
        assert len(self.models) > 0
        return self.models[self.master_model_idx].named_parameters()

    def get_model(self):
        assert len(self.models) > 0
        return self.models[self.master_model_idx]

    def train(self, is_train: bool = True):
        super().train(is_train)
        for model in self.models:
            model.train(is_train)

    def lr_scheduler_step(self):
        for scheduler in self.schedulers:
            scheduler.step()


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
        self.dataset_train = datasets.MNIST('../data', train=False, download=True, transform=transform)
        self.dataset_val = datasets.MNIST('../data', train=False, transform=transform)
        shrink_dataset = True
        if shrink_dataset:
            train_data_size = 2000
            val_data_size = 1000
            self.dataset_train.data = self.dataset_train.data[:train_data_size]
            self.dataset_train.targets = self.dataset_train.targets[:train_data_size]
            self.dataset_val.data = self.dataset_val.data[:val_data_size]
            self.dataset_val.targets = self.dataset_val.targets[:val_data_size]
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_val, **test_kwargs)

        def net_factory():
            return Net()

        def optimizer_factory(params):
            return optim.Adadelta(params, lr=args.lr)

        self.loss_func = partial(F.nll_loss, reduction="mean")

        def lr_scheduler_factory(optimizer):
            return StepLR(optimizer, step_size=1, gamma=args.gamma)

        self.reference_model = ReferenceModel(net_factory, optimizer_factory, lr_scheduler_factory, self.loss_func)

        num_replicas = args.num_replicas
        self.dataparallel_model = DataparallelModel(net_factory, optimizer_factory, lr_scheduler_factory,
                                                    self.loss_func, num_replicas,
                                                    reference_model=self.reference_model.get_model())

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            self.test()
            self.reference_model.lr_scheduler_step()
            self.dataparallel_model.lr_scheduler_step()

        if self.args.save_model:
            torch.save(self.reference_model.get_model().state_dict(), "mnist_cnn_ref.pt")
            torch.save(self.dataparallel_model.get_model().state_dict(), "mnist_cnn_ref.pt")

    def train_epoch(self, epoch):
        self.reference_model.train(True)
        self.dataparallel_model.train(True)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            step_info_ref = self.reference_model.step(data, target)
            ref_loss = step_info_ref["loss"]

            step_info_dp = self.dataparallel_model.step(data, target)
            dp_loss = step_info_dp["loss"]

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRef loss: {:.6f}\tDP loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), ref_loss, dp_loss))

    def test(self):
        self.dataparallel_model.train(False)

        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            step_info_dp = self.dataparallel_model.step(data, target, no_grad=True)
            test_loss += step_info_dp["loss"]  # sum up batch loss
            pred = step_info_dp["pred"].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


def parse_args(external_args=None):
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
    parser.add_argument('--num-replicas', type=int, default=4, metavar='N',
                        help='number of dataparallel replicas (default: 4)')
    if external_args is not None:
        args = parser.parse_args(external_args)
    else:
        args = parser.parse_args()
    return args


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
