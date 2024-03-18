import numpy as np
import torch


class EarlyStopping:

    def __init__(self, patience=100, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs = 0
        self.F1 = 0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs, F1, F2, F3, F4, model, modelname, str):

        # score = (accs + F1 + F2 + F3 + F4) / 5
        score = accs

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.save_checkpoint(val_loss, model, modelname, str)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("BEST Accuracy: {:.3f}|NR F1: {:.3f}|FR F1: {:.3f}|TR F1: {:.3f}|UR F1: {:.3f}"
                      .format(self.accs, self.F1, self.F2, self.F3, self.F4))
        else:
            self.best_score = score
            self.accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
            self.save_checkpoint(val_loss, model, modelname, str)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, modelname, str):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(), modelname + str + '.m')
        self.val_loss_min = val_loss
