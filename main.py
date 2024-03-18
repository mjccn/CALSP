
import os
import sys
sys.path.append(os.getcwd())
from Process.process import *
from others.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from others.evaluate import *
import random
from model import *
import numpy as np


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


setup_seed(2022)

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='hard_fc1.'): # T15: epsilon = 0.2
        for name, param in self.model.GCN_Net.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = th.norm(param.grad)
                if norm != 0 and not th.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='hard_fc1.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def train_GCN(x_test, x_train, lr, weight_decay, patience, n_epochs, batchsize, dataname):

    model = CALSP(768, 64, 64).to(device)

    fgm = FGM(model)
    for para in model.GCN_Net.hard_fc1.parameters():
        para.requires_grad = False

    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)


    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)


    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadData(dataname, x_train, x_test, droprate=0.4)  # T15 droprate = 0.1  T16 0.4
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5, drop_last=False)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5, drop_last=False)
        tqdm_train_loader = tqdm(train_loader)

        avg_loss = []
        avg_acc = []
        batch_idx = 0
        NUM = 1
        beta = 0.001

        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            predict, contrastive_loss = model(Batch_data)
            lable_train = th.cat((Batch_data.y1, Batch_data.y1, Batch_data.y1), 0)
            loss_cross = F.nll_loss(predict, lable_train)
            loss = loss_cross + beta * contrastive_loss
            avg_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            fgm.attack()
            predict, contrastive_loss = model(Batch_data)
            lable_train = th.cat((Batch_data.y1, Batch_data.y1, Batch_data.y1), 0)
            loss_cross = F.nll_loss(predict, lable_train)
            loss_adv = loss_cross + beta * contrastive_loss
            loss_adv.backward()
            fgm.restore()

            optimizer.step()


            _, pred = predict.max(dim=-1)
            correct = pred.eq(lable_train).sum().item()
            train_acc = correct / len(lable_train)
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1
            NUM += 1
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
            temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
            temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
            temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()

        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            predict_test, _ = model(Batch_data)
            lable_test = th.cat((Batch_data.y1, Batch_data.y1, Batch_data.y1), 0)
            val_loss = F.nll_loss(predict_test, lable_test)
            temp_val_losses.append(val_loss.item())
            _, val_pred = predict_test.max(dim=1)
            correct = val_pred.eq(lable_test).sum().item()
            val_acc = correct / len(lable_test)

            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, lable_test)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
                temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
                temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
                temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(avg_loss),
                                                                            np.mean(temp_val_losses),
                                                                            np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)

        if epoch > 25:
            early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                           np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'CALSP', dataname)
        accs = np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return accs, F1, F2, F3, F4


##---------------------------------main---------------------------------------
if __name__ == '__main__':
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    lr = 0.0005
    weight_decay = 1e-4
    patience = 10
    n_epochs = 200
    batchsize = 120

    datasetname = ''
    data_path = ''
    label_path = ''

    test_accs = []
    NR_F1 = []  # NR
    FR_F1 = []  # FR
    TR_F1 = []  # TR
    UR_F1 = []  # UR

    fold0_x_test, fold0_x_train, \
        fold1_x_test, fold1_x_train, \
        fold2_x_test, fold2_x_train, \
        fold3_x_test, fold3_x_train, \
        fold4_x_test, fold4_x_train = load5foldData(datasetname, data_path, label_path)

    print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train))
    print('fold1 shape: ', len(fold1_x_test), len(fold1_x_train))
    print('fold2 shape: ', len(fold2_x_test), len(fold2_x_train))
    print('fold3 shape: ', len(fold3_x_test), len(fold3_x_train))
    print('fold4 shape: ', len(fold4_x_test), len(fold4_x_train))

    accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(fold0_x_test, fold0_x_train, lr, weight_decay, patience, n_epochs,
                                              batchsize, datasetname)
    accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(fold1_x_test, fold1_x_train, lr, weight_decay, patience, n_epochs,
                                              batchsize, datasetname)
    accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(fold2_x_test, fold2_x_train, lr, weight_decay, patience, n_epochs,
                                              batchsize, datasetname)
    accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(fold3_x_test, fold3_x_train, lr, weight_decay, patience, n_epochs,
                                              batchsize, datasetname)
    accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(fold4_x_test, fold4_x_train, lr, weight_decay, patience, n_epochs,
                                              batchsize, datasetname)
    test_accs.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
    NR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
    print(
        "AVG_result: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(sum(test_accs), sum(NR_F1),
                                                                                            sum(FR_F1), sum(TR_F1),
                                                                                            sum(UR_F1)))
