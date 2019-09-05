from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable

from math import log10


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        
        for epoch in range(num_epochs):
            total_train = 0
            correct_train = 0
            running_loss = 0.0

            for i, data in enumerate(train_loader, 0):
                
                inputs = torch.autograd.Variable(data[0])
                target = torch.autograd.Variable(data[1]).long()
                
                optim.zero_grad()
                
                output = model.forward(inputs)
                _, predicted = torch.max(output.data, 1)
                loss = self.loss_func(output, target)
                
                loss.backward()
                optim.step()
                
                self.train_loss_history.append(loss.item())
                running_loss += loss.item()
                
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
                acc_train = correct_train/total_train
                self.train_acc_history.append(acc_train)
                
            print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' %(epoch + 1, num_epochs, acc_train, running_loss))
            
            '''
            correct_val = 0
            total_val = 0
            running_loss_val = 0.0
            with torch.no_grad():
                for _, batch in enumerate(val_loader, 0):
                    inputs_v = torch.autograd.Variable(batch[0])
                    target_v = torch.autograd.Variable(batch[1]).long()

                    prediction = model.forward(inputs_v)
                    loss = self.loss_func(output, target_v)
                    self.val_loss_history.append(loss.item())
                    running_loss_val += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    
                    total_val += target_v.size(0)
                    correct_val += (predicted == target_v).sum().item()
                    acc_val = correct_val/total_val
                    self.val_acc_history.append(acc_val)
                    
            print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' %(epoch + 1, num_epochs, acc_val, running_loss_val))
            '''
        #self.model.params = self.best_params
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
