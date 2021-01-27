import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import parameters as par
from Models import Classifier_1
from Custom_Loss_function import Loss_fairness_regularization
from Dataset_maker import Dataset_maker, Dataset_maker_clf_distribution
from functions.classifier_metrics import balanced_accuracy_gender, accuracy_calculate_gender, EO_evaluation

DATA_URL = par.ENCODED_DATA
ClASSIFIER_SAVE_PATH = par.CLASSIFIER_SAVE_PATH
RECORD_PATH = par.CLASSIFIER_RECORD_PATH
TEST_SAVE_PATH = par.CLASSIFIER_TESTSET_PATH

# hyper parameters
EPOCH = par.CLF_EPOCH
LR = par.CLF_LR

# ======= make new folder ==========
if os.path.exists(ClASSIFIER_SAVE_PATH) is False:
    os.mkdir(ClASSIFIER_SAVE_PATH)
if os.path.exists(RECORD_PATH) is False:
    os.mkdir(RECORD_PATH)

# ========= make dataset from encoded dataset ==========
train_db, val_db, test_df = Dataset_maker_clf_distribution(DATA_URL, [0.8, 0.1, 0.1])
model = Classifier_1()
# save test dataframe for test in next step
test_df.to_csv(TEST_SAVE_PATH, index=False, sep=',')

# =================== loss function ====================
Lossfunc = nn.BCELoss()
# Lossfunc_val = nn.BCELoss()
Lossfunc = Loss_fairness_regularization()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
classification_loader_train = Data.DataLoader(dataset=train_db, batch_size=len(train_db), shuffle=True, num_workers=0)
classification_loader_val = Data.DataLoader(dataset=val_db, batch_size=len(val_db), shuffle=True, num_workers=0)

writer_classifier = SummaryWriter(RECORD_PATH)
# train classifier with classifier_1
for iter in range(EPOCH):
    # ========================train ============================
    for i, (data, label) in enumerate(classification_loader_train):
        input_data = Variable(data)
        # ====== bce loss ======
        # gt = Variable(label)
        # true_label = gt[:][:, 0].clone().detach().view(len(train_db), 1)
        # true_label = Variable(true_label)
        # === regularization loss =====
        true_label = Variable(label)
        # =======forward==============
        pred_label = model(input_data)
        loss, loss_item = Lossfunc(pred_label, true_label)
        # =======backward=============
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ======================validation===========================
    for j, (data_val, label_val) in enumerate(classification_loader_val):
        input_data_val = Variable(data_val)
        # ===== bce loss =====
        # gt_val = Variable(label_val)
        # true_label_val = gt_val[:][:, 0].clone().detach().view(len(val_db), 1)
        # =====regularization loss====
        true_label_val = Variable(label_val)
        pred_label_val = model(input_data_val)
        loss_val, _ = Lossfunc(pred_label_val, true_label_val)
        # loss_val = Lossfunc_val(pred_label_val, true_label_val)
    acc_val, _, _ = accuracy_calculate_gender(val_db, model)
    _, EO = EO_evaluation(val_db, model)
    print('EO_val:', EO)
    # ==========log==============
    print('epoch :[{}/{}], loss:{:.5f}'.format(iter+1, EPOCH, loss.data.item()))
    writer_classifier.add_scalar('Train/Loss', loss.data.item(), iter + 1)
    writer_classifier.add_scalar('Val/Acc', acc_val, iter + 1)
    writer_classifier.add_scalar('Val/Loss', loss_val, iter + 1)
    writer_classifier.add_scalar('Val/EO', EO, iter + 1)
    # ==========save model=======
    torch.save(model, ClASSIFIER_SAVE_PATH + '/EO_Classifier_%s_%s.pkl' % (iter + 1, EPOCH))

writer_classifier.close()