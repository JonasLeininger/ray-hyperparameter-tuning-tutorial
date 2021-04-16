import os
import sys
import time
import numpy as np
from ray import tune
from torch.utils.tensorboard import SummaryWriter
from filelock import FileLock
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
from torchvision import datasets, transforms

from hyperparam_tune.model.resnet18 import ResNet18
from hyperparam_tune.model import resnet_template
from hyperparam_tune.utils.annotations import override
from hyperparam_tune.config.config import Config
from hyperparam_tune.utils.meters import AverageMeter, ProgressMeter
from hyperparam_tune.utils.bit_hyperrule import get_resolution_from_dataset

cudnn.benchmark = True

class Trainer(tune.Trainable):
    _name = "ResNet18Trainer"

    @override(tune.Trainable)
    def setup(self, config):
        self.config = config
        self.device, self.multi_gpu = self._get_device_info()
        self.train_loader, self.val_loader, self.test_loader = self.load_data()
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), 
            lr=0.003, momentum=0.9)
        
        self.writer = SummaryWriter(log_dir=config['tensorboard_log_dir'])
        self.epoch = 0
        self.debug = False
        # self.mixup_alpha = 0.1 if (len(self.train_loader) > 20000) else 0.0
        self.mixup_alpha = 0.0
    
    def _train(self): 
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        accuracies = AverageMeter('Acc', ':6.3f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, accuracies],
            prefix="Epoch: [{}]".format(self.epoch))
        
        self.model.train()
        end = time.time()

        for i, (images, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            data = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            mixup_l = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1


            if self.mixup_alpha > 0:
                x, y_a, y_b = self._mixup_data(data, target, mixup_l)
                output = self.model(x)
                loss = self._mixup_criterion(criterion, output, y_a, y_b, mixup_l)

            else: 
                output = self.model(data)
                loss = self.criterion(output, target)

            acc1 = self.accuracy(output, target, topk=(1,))

            losses.update(loss.item(), images.size(0))
            accuracies.update(acc1[0].item(), data.size(0))

            self.optimizer.zero_grad()
            (loss / self.config['batch_split']).backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.debug: 
                if i % 50 == 0:
                    progress.display(i)

        return losses.avg, accuracies.avg
    

    def _val(self, mode='val'):

        dataloader = self.val_loader
        prefix = 'Eval'

        if mode == 'test':
            dataloader = self.test_loader
            prefix = 'Test'


        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        accuracies = AverageMeter('Acc', ':6.2f')
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, losses, accuracies],
            prefix=prefix + ': ')
        
        self.model.eval()

        with torch.no_grad(): 
            end = time.time()
            for i, (images, target) in enumerate(dataloader):
                data, target = images.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                acc1 = self.accuracy(output, target, topk=(1,))
                accuracies.update(acc1[0].item(), data.size(0))
                losses.update(loss.item(), data.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if self.debug:
                    if i % 50 == 0:
                        progress.display(i)

        return losses.avg, accuracies.avg


    @override(tune.Trainable)
    def step(self):

        self.epoch += 1

        ## Training
        train_loss, train_acc = self._train()

        ## Evaluation
        val_loss, val_acc = self._val(mode='val')

        ## Test
        test_loss, test_acc = self._val(mode='test')

        self._weights_for_tensorboard_log(self.epoch)

        return {
            "loss": train_loss, 
            "accuracy": train_acc, 
            "val_loss": val_loss,
            "val_accuracy": val_acc, 
            "test_loss": test_loss,
            "test_accuracy": test_acc
            }
    
    def load_data(self, data_dir='./torch/cifar10'):

        use_cuda = "cuda" in self.device
        kwargs = {"num_workers": self.config['num_workers'], "pin_memory": True} if use_cuda else {}
        batch_size = self.config['batch_size']

        dataset = self.config['dataset']
        precrop, crop = get_resolution_from_dataset(dataset)

        train_tx = transforms.Compose([
            transforms.Resize((precrop, precrop)),
            transforms.RandomCrop((crop, crop)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ])

        val_tx = transforms.Compose([
            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        with FileLock("./data.lock"):

            if dataset == 'cifar10':
                trainset = datasets.CIFAR10(
                    root=data_dir, train=True, download=True, transform=train_tx)
                testset = datasets.CIFAR10(
                    root=data_dir, train=False, download=True, transform=val_tx)
            elif dataset == 'cifar100':
                trainset = datasets.CIFAR100(
                    root=data_dir, train=True, download=True, transform=train_tx)
                testset = datasets.CIFAR100(
                    root=data_dir, train=False, download=True, transform=val_tx)
            # elif dataset == 'freiburg'
            #    trainset = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
            #    validset = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
            
            micro_batch_size = self.config['batch_size'] // self.config['batch_split']

            val_abs = int(len(trainset) * 0.8)
            train_subset, val_subset = random_split(trainset, [val_abs, len(trainset) - val_abs])

            train_loader = torch.utils.data.DataLoader(train_subset, micro_batch_size, shuffle=True, **kwargs)
            val_loader = torch.utils.data.DataLoader(val_subset, micro_batch_size, shuffle=False, **kwargs)
            test_loader = torch.utils.data.DataLoader(testset, micro_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader
    
    def _get_device_info(self): 
        multi_gpu = False
        if torch.cuda.is_available(): 
            device = "cuda:0"
            if torch.cuda.device_count() > 1: 
                multi_gpu = True
        else: 
            device = "cpu"
        
        return device, multi_gpu

    def _create_model(self):

        model_type = self.config['model_type']
        
        if model_type == 'resnet18':
            model = ResNet18(self.config)
        else:
            model = resnet_template.KNOWN_MODELS[model_type](
                head_size=self.config['num_classes'], zero_head=True
            )

            home = str(Path.home())
            model_path = os.path.join(home, self.config['bit_model_dir'], model_type)
            model.load_from(np.load(model_path + '.npz'))

        if self.multi_gpu: 
            model = nn.DataParallel(model)
        
        ## INSERT CODE FOR RESUME FINE_TUNING HERE 

        model.to(self.device)
        return model

    def _weights_for_tensorboard_log(self, train_count):
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(
                "model-weights" + name, param.clone().cpu().data.numpy(), train_count
            )
    
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def _mixup_data(self, x, y, l):
        """Returns mixed inputs, pairs of targets, and lambda"""
        indices = torch.randperm(x.shape[0]).to(x.device)

        mixed_x = l * x + (1 - l) * x[indices]
        y_a, y_b = y, y[indices]
        return mixed_x, y_a, y_b


    def _mixup_criterion(self, criterion, pred, y_a, y_b, l):
        return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)
        
    @override(tune.Trainable)
    def _save(self, checkpoint_path):
        print('Saving model at {}'.format(checkpoint_path))
        checkpoint_path_model_name = os.path.join(checkpoint_path, "model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path_model_name)

        return checkpoint_path_model_name

    @override(tune.Trainable)
    def _restore(self, checkpoint_path):
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print('Restored model from {}'.format(checkpoint_path))

def main(): 
    config = Config(config_file="src/hyperparam_tune/config/config_local.yaml").config
    print(config)
    trainer = Trainer(config=config)
    for i in range(1):
        result_dict = trainer.step()
        # print(result_dict)
    
    ckpt_model_name = trainer._save('ckpt/')

    trainer.model = trainer._create_model()
    trainer._val(mode='test')

    trainer._restore(ckpt_model_name)
    trainer._val(mode='test')



if __name__ == "__main__":
    main()
    