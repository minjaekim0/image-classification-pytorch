import math
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import torch


class Trainer:
    def __init__(self, train_loader, test_loader, num_epochs, model, criterion, optimizer, scheduler):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_batches_train = len(self.train_loader)
        self.num_batches_valid = len(self.test_loader)

        self.device = torch.device('cuda')
        self.model.to(self.device)

        self.train_loss_list = []
        self.train_acc_list = []
        self.valid_loss_list = []
        self.valid_acc_list = []
        self.last_save_path = None

        self.start = time.time()


    def train_one_epoch(self, epoch):
        """train model for one epoch"""

        train_loss = 0
        total = 0
        correct = 0

        for _batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # _batch_idx starts from 0
            # batch_idx = _batch_idx + 1 starts from 1
            batch_idx = _batch_idx + 1

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += torch.eq(predicted, targets).sum().item()

            actual_train_loss = train_loss / batch_idx

            now = time.time()
            print('\r', end='')
            print(self.status_bar(now-self.start, epoch, 'train', batch_idx,
                                  actual_train_loss, correct / total), end='  ')
        print('\n', end='')

        return {
            'epoch': epoch,
            'loss': train_loss / self.num_batches_train,
            'acc': correct / total
        }


    def valid_one_epoch(self, epoch):
        """valid model for one epoch"""

        self.model.eval()
        valid_loss = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for _batch_idx, (inputs, targets) in enumerate(self.test_loader):
                # _batch_idx starts from 0
                # batch_idx = _batch_idx + 1 starts from 1
                batch_idx = _batch_idx + 1

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += torch.eq(predicted, targets).sum().item()

                actual_valid_loss = valid_loss / batch_idx

                now = time.time()
                print('\r', end='')
                print(self.status_bar(now-self.start, epoch, 'valid', batch_idx,
                                      actual_valid_loss, correct / total), end='  ')
        print('\n\n', end='')

        return {
            'epoch': epoch,
            'loss': valid_loss / self.num_batches_valid,
            'acc': correct / total
        }


    def save(self, epoch, acc):
        """model save"""

        # remove last saved model
        if self.last_save_path:
            os.remove(self.last_save_path)

        state = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'acc': acc
        }
        path = f'saved_models/best_{self.model}' \
               f'_{datetime.now().strftime("%Y%m%d_%H%M")}.pth'
        torch.save(state, path)

        self.last_save_path = path


    def run(self):
        """execute overall training process"""

        for i in range(1, self.num_epochs+1):
            train_result = self.train_one_epoch(i)
            valid_result = self.valid_one_epoch(i)
            self.scheduler.step()
            self.train_loss_list.append(train_result['loss'])
            self.train_acc_list.append(train_result['acc'])
            self.valid_loss_list.append(valid_result['loss'])
            self.valid_acc_list.append(valid_result['acc'])

            last_acc = self.valid_acc_list[-1]
            if last_acc == max(self.valid_acc_list):
                self.save(i, last_acc)


    def plot(self):
        """plot train/valid loss and accuracy vs epochs"""

        sns.set_style("darkgrid")
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        x = list(range(1, self.num_epochs + 1))
        sns.lineplot(x=x, y=self.train_loss_list, label='train_loss', ax=ax1)
        sns.lineplot(x=x, y=self.valid_loss_list, label='valid_loss', ax=ax1)
        sns.lineplot(x=x, y=self.train_acc_list, label='train_acc', ax=ax2)
        sns.lineplot(x=x, y=self.valid_acc_list, label='valid_acc', ax=ax2)

        ax1.set_ylabel('loss')
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('epoch')

        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        ax1.legend(ax1.get_legend_handles_labels()[1], loc='best')
        ax2.legend(ax2.get_legend_handles_labels()[1], loc='best')

        fig.savefig(f'plots/loss_acc_plot_{self.model}'
                    f'_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        fig.show()


    def status_bar(self, time_elapsed, epoch, train_or_valid, batch_idx, loss, acc, length=50):
        """show status at train_one_epoch & valid_one_epoch"""
        if train_or_valid == 'train':
            num_batches = self.num_batches_train
        else:
            num_batches = self.num_batches_valid

        rate = batch_idx / num_batches
        num_equals = math.floor(length * rate) - 1
        num_dots = length - num_equals - 1

        status = f'[{time_elapsed // 60:3.0f}m {time_elapsed % 60:5.2f}s | ' \
                 f'epoch: {epoch}/{self.num_epochs} | {train_or_valid:5s}]' \
                 f'[{"=" * num_equals}>{"." * num_dots}]' \
                 f'[loss:{loss:6.3f} | acc:{acc:8.3%}]'

        return status