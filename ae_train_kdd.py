import os
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import pandas as pd

import parameters as par
from Models import MyAutoencoder
from Custom_Loss_function import MultiLoss, MultiLoss_JSD
from Dataset_maker import Dataset_maker
from functions.utils import dataset_splite, plot_loss

URL = par.KDD_URL
RESULT_SAVE = par.PLOT_RESULT
SAVE_PATH = par.AE_MODEL_SAVE_PATH

# hyper parameters
EPOCH = par.AE_EPOCH
BATCH_SIZE = par.AE_BATCH_SIZE
LR = par.AE_LR


def model_train(train_loader, val_loader, iteration, learning_rate, loss_record, save_path):
    model = MyAutoencoder()
    loss_func = MultiLoss_JSD()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=learning_rate)

    for iter in range(iteration):
        for i, (data, label) in enumerate(train_loader):
            data_tensor = data
            data_tensor = Variable(data_tensor)
            # ==============forward==============
            data_encoded, data_decoded = model(data_tensor)
            loss, loss_items = loss_func(data_encoded, data_decoded, data_tensor, label, len(data_tensor))
            # ==============backward=============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ==============validation===============
        for j, (data_val, label_val) in enumerate(val_loader):
            data_tensor_val = data_val
            data_tensor_val = Variable(data_tensor_val)
            data_encoded_val, data_decoded_val = model(data_tensor_val)
            loss_val, loss_items_val = loss_func(data_encoded_val, data_decoded_val, data_tensor_val, label_val,
                                                 len(data_tensor_val))
        # ==============save model===============
        torch.save(model, save_path + '/KLD_AE_%s_%s.pkl' % (iter + 1, EPOCH))
        # =============== log =====================
        loss_record = loss_record.append({'Multi_loss_train': float(loss.data.item()),
                                          'mse_loss': float(loss_items[0].data.item()),
                                          'crossEntropy_loss': float(loss_items[1].data.item()),
                                          'KLD': float(loss_items[2].item()),
                                          'Multi_loss_val': float(loss_val.data.item())
                                          }, ignore_index=True)
        print('epoch [{}/{}], train loss:{:.4f}, train KLD:{:.4f}, val loss:{:.4f}'.format(iter + 1, iteration,
                                                                                           loss.data.item(),
                                                                                           loss_items[2].item(),
                                                                                           loss_val.data.item()))

    return loss_record, model


if __name__ == '__main__':
    # ===== make dataset =====
    MyDataset = Dataset_maker(URL, mode='c')
    print(len(MyDataset))
    train_db, val_db, test_db = dataset_splite(MyDataset, 179570, 19953, 19953)

    train_loader = Data.DataLoader(dataset=train_db,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=0)
    val_loader = Data.DataLoader(dataset=val_db,
                                 batch_size=len(val_db),
                                 shuffle=False,
                                 num_workers=0)
    test_loader = Data.DataLoader(dataset=test_db,
                                  batch_size=len(test_db),
                                  shuffle=False,
                                  num_workers=0)
    # ===== train model ======
    if os.path.exists(SAVE_PATH) is False:
        os.mkdir(SAVE_PATH)
    loss_record = pd.DataFrame(
        columns=('Multi_loss_train', 'mse_loss', 'crossEntropy_loss', 'KLD', 'Multi_loss_val'))
    loss_record, model = model_train(train_loader, val_loader, EPOCH, LR, loss_record, SAVE_PATH)
    plot_loss(loss_record, RESULT_SAVE, EPOCH)

