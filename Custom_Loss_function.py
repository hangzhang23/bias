"""
the multi loss function is made up with three parts:
restruction loss, mse loss and crossEntropy loss
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from functions.classifier_metrics import Equalized_odds
from functions.utils import js_distance, KL_divergence
import parameters as par


# ============== autoencoder loss function ===============
class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()

    def decoded_crossentropy(self, data_decoded, start_index, end_index):
        return torch.index_select(data_decoded, 1,
                                  torch.linspace(start_index, end_index, (end_index - start_index + 1)).long())

    def true_crossentropy(self, data_true, start_index, end_index, batch_size):
        data_true_index = torch.ones(batch_size, 1)
        data_true_slice = torch.index_select(data_true, 1, torch.linspace(start_index, end_index,
                                                                          (end_index - start_index + 1)).long())
        for i in range(batch_size):
            data_true_index[i] = data_true_slice[i].nonzero()[0]
        return data_true_index.long().squeeze(1)

    def forward(self, data_decoded, data_true, batch_size):
        loss_item = []

        # === age loss ===
        age_loss = F.mse_loss(data_decoded[:][:, 0], data_true[:][:, 0])
        # === class of worker loss ===
        cls_wor_decoded = self.decoded_crossentropy(data_decoded, 1, 9)
        cls_wor_true = self.true_crossentropy(data_true, 1, 9, batch_size)
        cls_wor_loss = F.cross_entropy(cls_wor_decoded, cls_wor_true)
        # === detailed industry recode loss ===
        detail_in_re_loss = F.mse_loss(data_decoded[:][:, 10], data_true[:][:, 10])
        # === detailed occupation recode loss ===
        detail_oc_re_loss = F.mse_loss(data_decoded[:][:, 11], data_true[:][:, 11])
        # === education loss ===
        edu_decoded = self.decoded_crossentropy(data_decoded, 12, 28)
        edu_true = self.true_crossentropy(data_true, 12, 28, batch_size)
        edu_loss = F.cross_entropy(edu_decoded, edu_true)
        # === wage per hour loss (29)===
        wage_loss = F.mse_loss(data_decoded[:][:, 0], data_true[:][:, 0])
        # === enroll in edu inst last wk loss (30-32)===
        enr_wk_decoded = self.decoded_crossentropy(data_decoded, 30, 32)
        enr_wk_true = self.true_crossentropy(data_true, 30, 32, batch_size)
        enr_wk_loss = F.cross_entropy(enr_wk_decoded, enr_wk_true)
        # === marital stat loss (33-39)===
        mar_sta_decoded = self.decoded_crossentropy(data_decoded, 33, 39)
        mar_sta_true = self.true_crossentropy(data_true, 33, 39, batch_size)
        mar_sta_loss = F.cross_entropy(mar_sta_decoded, mar_sta_true)
        # === major industry code loss (40-63)===
        maj_ind_decoded = self.decoded_crossentropy(data_decoded, 40, 63)
        maj_ind_true = self.true_crossentropy(data_true, 40, 63, batch_size)
        maj_ind_loss = F.cross_entropy(maj_ind_decoded, maj_ind_true)
        # === major occupation code loss (64-78)===
        maj_occ_decoded = self.decoded_crossentropy(data_decoded, 64, 78)
        maj_occ_true = self.true_crossentropy(data_true, 64, 78, batch_size)
        maj_occ_loss = F.cross_entropy(maj_occ_decoded, maj_occ_true)
        # === race loss (79-83)===
        race_decoded = self.decoded_crossentropy(data_decoded, 79, 83)
        race_true = self.true_crossentropy(data_true, 79, 83, batch_size)
        race_loss = F.cross_entropy(race_decoded, race_true)
        # === hispanic origin loss (84-93)===
        his_ori_decoded = self.decoded_crossentropy(data_decoded, 84, 93)
        his_ori_true = self.true_crossentropy(data_true, 84, 93, batch_size)
        his_ori_loss = F.cross_entropy(his_ori_decoded, his_ori_true)
        # === sex loss (94-95)===
        sex_decoded = self.decoded_crossentropy(data_decoded, 94, 95)
        sex_true = self.true_crossentropy(data_true, 94, 95, batch_size)
        sex_loss = F.cross_entropy(sex_decoded, sex_true)
        # === member of a labor union loss (96-98)===
        mem_lab_decoded = self.decoded_crossentropy(data_decoded, 96, 98)
        mem_lab_true = self.true_crossentropy(data_true, 96, 98, batch_size)
        mem_lab_loss = F.cross_entropy(mem_lab_decoded, mem_lab_true)
        # === reason for unemployment loss (99-104)===
        rea_une_decoded = self.decoded_crossentropy(data_decoded, 99, 104)
        rea_une_true = self.true_crossentropy(data_true, 99, 104, batch_size)
        rea_une_loss = F.cross_entropy(rea_une_decoded, rea_une_true)
        # === full or part time employment stat loss (105-112)===
        ful_par_decoded = self.decoded_crossentropy(data_decoded, 105, 112)
        ful_par_true = self.true_crossentropy(data_true, 105, 112, batch_size)
        ful_par_loss = F.cross_entropy(ful_par_decoded, ful_par_true)
        # === capital gains loss (113)===
        cap_gain_loss = F.mse_loss(data_decoded[:][:, 113], data_true[:][:, 113])
        # === capital losses loss (114)===
        cap_loss_loss = F.mse_loss(data_decoded[:][:, 114], data_true[:][:, 114])
        # === dividends from stocks loss (115)===
        div_stock_loss = F.mse_loss(data_decoded[:][:, 115], data_true[:][:, 115])
        # === tax filer stat loss (116-121)===
        tax_decoded = self.decoded_crossentropy(data_decoded, 116, 121)
        tax_true = self.true_crossentropy(data_true, 116, 121, batch_size)
        tax_loss = F.cross_entropy(tax_decoded, tax_true)
        # === region of previous residence loss (122-127)===
        reg_res_decoded = self.decoded_crossentropy(data_decoded, 122, 127)
        reg_res_true = self.true_crossentropy(data_true, 122, 127, batch_size)
        reg_res_loss = F.cross_entropy(reg_res_decoded, reg_res_true)
        # === detailed household and family stat loss (128-150)===
        fam_sta_decoded = self.decoded_crossentropy(data_decoded, 128, 150)
        fam_sta_true = self.true_crossentropy(data_true, 128, 150, batch_size)
        fam_sta_loss = F.cross_entropy(fam_sta_decoded, fam_sta_true)
        # === detailed household summary in household loss (151-158)===
        sum_hou_decoded = self.decoded_crossentropy(data_decoded, 151, 158)
        sum_hou_true = self.true_crossentropy(data_true, 151, 158, batch_size)
        sum_hou_loss = F.cross_entropy(sum_hou_decoded, sum_hou_true)
        # === num persons worked for employer loss (159)===
        num_emp_loss = F.mse_loss(data_decoded[:][:, 159], data_true[:][:, 159])
        # === family members under 18 loss (160-164)===
        fam_u18_decoded = self.decoded_crossentropy(data_decoded, 160, 164)
        fam_u18_true = self.true_crossentropy(data_true, 160, 164, batch_size)
        fam_u18_loss = F.cross_entropy(fam_u18_decoded, fam_u18_true)
        # === own business of self employed loss (165)===
        own_bus_loss = F.mse_loss(data_decoded[:][:, 165], data_true[:][:, 165])
        # === veterans benefits loss (166)===
        vet_ben_loss = F.mse_loss(data_decoded[:][:, 166], data_true[:][:, 166])
        # === weeks worked in year loss (167)===
        wek_yea_loss = F.mse_loss(data_decoded[:][:, 167], data_true[:][:, 167])

        Multi_loss = age_loss + cls_wor_loss + detail_in_re_loss + detail_oc_re_loss + edu_loss + wage_loss + \
                     enr_wk_loss + mar_sta_loss + maj_ind_loss + maj_occ_loss + race_loss + his_ori_loss + sex_loss + \
                     mem_lab_loss + rea_une_loss + ful_par_loss + cap_gain_loss + cap_loss_loss + div_stock_loss + \
                     tax_loss + reg_res_loss + fam_sta_loss + sum_hou_loss + num_emp_loss + fam_u18_loss + own_bus_loss + \
                     vet_ben_loss + wek_yea_loss

        mse_Loss = age_loss + detail_in_re_loss + detail_oc_re_loss + wage_loss + cap_gain_loss + cap_loss_loss + \
                   div_stock_loss + num_emp_loss + own_bus_loss + vet_ben_loss + wek_yea_loss

        ce_Loss = cls_wor_loss + edu_loss + enr_wk_loss + mar_sta_loss + maj_ind_loss + maj_occ_loss + race_loss + \
                  his_ori_loss + sex_loss + mem_lab_loss + rea_une_loss + ful_par_loss + tax_loss + reg_res_loss + \
                  fam_sta_loss + sum_hou_loss + fam_u18_loss

        loss_item.append(mse_Loss)
        loss_item.append(ce_Loss)

        return Multi_loss, loss_item


class MultiLoss_JSD(nn.Module):
    def __init__(self):
        super(MultiLoss_JSD, self).__init__()

    def decoded_crossentropy(self, data_decoded, start_index, end_index):
        return torch.index_select(data_decoded, 1,
                                  torch.linspace(start_index, end_index, (end_index - start_index + 1)).long())

    def true_crossentropy(self, data_true, start_index, end_index, batch_size):
        data_true_index = torch.ones(batch_size, 1)
        data_true_slice = torch.index_select(data_true, 1, torch.linspace(start_index, end_index,
                                                                          (end_index - start_index + 1)).long())
        for i in range(batch_size):
            data_true_index[i] = data_true_slice[i].nonzero()[0]
        return data_true_index.long().squeeze(1)

    def forward(self, data_encoded, data_decoded, data_true, label_true, batch_size):
        loss_item = []

        # === age loss ===
        age_loss = F.mse_loss(data_decoded[:][:, 0], data_true[:][:, 0])
        # === class of worker loss （9）===
        cls_wor_decoded = data_decoded[:, 1:10]  # self.decoded_crossentropy(data_decoded, 1, 9) #
        cls_wor_true = data_true[:, 1:10].nonzero()[:, 1]  # self.true_crossentropy(data_true, 1, 9, batch_size) #
        cls_wor_loss = F.cross_entropy(cls_wor_decoded, cls_wor_true)
        # === detailed industry recode loss （1）===
        detail_in_re_loss = F.mse_loss(data_decoded[:][:, 10], data_true[:][:, 10])
        # === detailed occupation recode loss （1）===
        detail_oc_re_loss = F.mse_loss(data_decoded[:][:, 11], data_true[:][:, 11])
        # === education loss （17）===
        edu_decoded = data_decoded[:, 12:29]  # self.decoded_crossentropy(data_decoded, 12, 28)
        edu_true = data_true[:, 12:29].nonzero()[:, 1]  # self.true_crossentropy(data_true, 12, 28, batch_size)
        edu_loss = F.cross_entropy(edu_decoded, edu_true)
        # === wage per hour loss (1)===
        wage_loss = F.mse_loss(data_decoded[:][:, 29], data_true[:][:, 29])
        # === enroll in edu inst last wk loss (3)===
        enr_wk_decoded = data_decoded[:, 30:33]  # self.decoded_crossentropy(data_decoded, 30, 32)
        enr_wk_true = data_true[:, 30:33].nonzero()[:, 1]  # self.true_crossentropy(data_true, 30, 32, batch_size)
        enr_wk_loss = F.cross_entropy(enr_wk_decoded, enr_wk_true)
        # === marital stat loss (7)===
        mar_sta_decoded = data_decoded[:, 33:40]  # self.decoded_crossentropy(data_decoded, 33, 39)
        mar_sta_true = data_true[:, 33:40].nonzero()[:, 1]  # self.true_crossentropy(data_true, 33, 39, batch_size)
        mar_sta_loss = F.cross_entropy(mar_sta_decoded, mar_sta_true)
        # === major industry code loss (24)===
        maj_ind_decoded = data_decoded[:, 40:64]  # self.decoded_crossentropy(data_decoded, 40, 63)
        maj_ind_true = data_true[:, 40:64].nonzero()[:, 1]  # self.true_crossentropy(data_true, 40, 63, batch_size)
        maj_ind_loss = F.cross_entropy(maj_ind_decoded, maj_ind_true)
        # === major occupation code loss (15)===
        maj_occ_decoded = data_decoded[:, 64:79]  # self.decoded_crossentropy(data_decoded, 64, 78)
        maj_occ_true = data_true[:, 64:79].nonzero()[:, 1]  # self.true_crossentropy(data_true, 64, 78, batch_size)
        maj_occ_loss = F.cross_entropy(maj_occ_decoded, maj_occ_true)
        # === race loss (5)===
        race_decoded = data_decoded[:, 79:84]  # self.decoded_crossentropy(data_decoded, 79, 83)
        race_true = data_true[:, 79:84].nonzero()[:, 1]  # self.true_crossentropy(data_true, 79, 83, batch_size)
        race_loss = F.cross_entropy(race_decoded, race_true)
        # === hispanic origin loss (10)===
        his_ori_decoded = data_decoded[:, 84:94]  # self.decoded_crossentropy(data_decoded, 84, 93)
        his_ori_true = data_true[:, 84:94].nonzero()[:, 1]  # self.true_crossentropy(data_true, 84, 93, batch_size)
        his_ori_loss = F.cross_entropy(his_ori_decoded, his_ori_true)
        # === sex loss (2)===
        sex_decoded = data_decoded[:, 94:96]  # self.decoded_crossentropy(data_decoded, 94, 95)
        sex_true = data_true[:, 94:96].nonzero()[:, 1]  # self.true_crossentropy(data_true, 94, 95, batch_size)
        sex_loss = F.cross_entropy(sex_decoded, sex_true)
        # === member of a labor union loss (3)===
        mem_lab_decoded = data_decoded[:, 96:99]  # self.decoded_crossentropy(data_decoded, 96, 98)
        mem_lab_true = data_true[:, 96:99].nonzero()[:, 1]  # self.true_crossentropy(data_true, 96, 98, batch_size)
        mem_lab_loss = F.cross_entropy(mem_lab_decoded, mem_lab_true)
        # === reason for unemployment loss (6)===
        rea_une_decoded = data_decoded[:, 99:105]  # self.decoded_crossentropy(data_decoded, 99, 104)
        rea_une_true = data_true[:, 99:105].nonzero()[:, 1]  # self.true_crossentropy(data_true, 99, 104, batch_size)
        rea_une_loss = F.cross_entropy(rea_une_decoded, rea_une_true)
        # === full or part time employment stat loss (8)===
        ful_par_decoded = data_decoded[:, 105:113]  # self.decoded_crossentropy(data_decoded, 105, 112)
        ful_par_true = data_true[:, 105:113].nonzero()[:, 1]  # self.true_crossentropy(data_true, 105, 112, batch_size)
        ful_par_loss = F.cross_entropy(ful_par_decoded, ful_par_true)
        # === capital gains loss (1)===
        cap_gain_loss = F.mse_loss(data_decoded[:][:, 113], data_true[:][:, 113])
        # === capital losses loss (1)===
        cap_loss_loss = F.mse_loss(data_decoded[:][:, 114], data_true[:][:, 114])
        # === dividends from stocks loss (1)===
        div_stock_loss = F.mse_loss(data_decoded[:][:, 115], data_true[:][:, 115])
        # === tax filer stat loss (6)===
        tax_decoded = data_decoded[:, 116:122]  # self.decoded_crossentropy(data_decoded, 116, 121)
        tax_true = data_true[:, 116:122].nonzero()[:, 1]  # self.true_crossentropy(data_true, 116, 121, batch_size)
        tax_loss = F.cross_entropy(tax_decoded, tax_true)
        # === region of previous residence loss (6)===
        reg_res_decoded = data_decoded[:, 122:128]  # self.decoded_crossentropy(data_decoded, 122, 127)
        reg_res_true = data_true[:, 122:128].nonzero()[:, 1]  # self.true_crossentropy(data_true, 122, 127, batch_size)
        reg_res_loss = F.cross_entropy(reg_res_decoded, reg_res_true)
        # === detailed household and family stat loss (23)===
        fam_sta_decoded = data_decoded[:, 128:151]  # self.decoded_crossentropy(data_decoded, 128, 150)
        fam_sta_true = data_true[:, 128:151].nonzero()[:, 1]  # self.true_crossentropy(data_true, 128, 150, batch_size)
        fam_sta_loss = F.cross_entropy(fam_sta_decoded, fam_sta_true)
        # === detailed household summary in household loss (8)===
        sum_hou_decoded = data_decoded[:, 151:159]  # self.decoded_crossentropy(data_decoded, 151, 158)
        sum_hou_true = data_true[:, 151:159].nonzero()[:, 1]  # self.true_crossentropy(data_true, 151, 158, batch_size)
        sum_hou_loss = F.cross_entropy(sum_hou_decoded, sum_hou_true)
        # === num persons worked for employer loss (1)===
        num_emp_loss = F.mse_loss(data_decoded[:][:, 159], data_true[:][:, 159])
        # === family members under 18 loss (5)===
        fam_u18_decoded = data_decoded[:, 160:165]  # self.decoded_crossentropy(data_decoded, 160, 164)
        fam_u18_true = data_true[:, 160:165].nonzero()[:, 1]  # self.true_crossentropy(data_true, 160, 164, batch_size)
        fam_u18_loss = F.cross_entropy(fam_u18_decoded, fam_u18_true)
        # === own business of self employed loss (1)===
        own_bus_loss = F.mse_loss(data_decoded[:][:, 165], data_true[:][:, 165])
        # === veterans benefits loss (1)===
        vet_ben_loss = F.mse_loss(data_decoded[:][:, 166], data_true[:][:, 166])
        # === weeks worked in year loss (1)===
        wek_yea_loss = F.mse_loss(data_decoded[:][:, 167], data_true[:][:, 167])


        # =========================== JS distance ===========================
        encoded_combined = torch.cat((data_encoded, label_true), 1)
        min_tensor = torch.min(encoded_combined, dim=0)[0]
        max_tensor = torch.max(encoded_combined, dim=0)[0]
        # split batch to male batch and female batch
        male_index = torch.squeeze(torch.nonzero(encoded_combined[:, 11] == 0))
        female_index = torch.squeeze(torch.nonzero(encoded_combined[:, 11] == 1))

        male_batch = torch.index_select(encoded_combined, 0, index=male_index)
        female_batch = torch.index_select(encoded_combined, 0, index=female_index)

        # find distribution of male and female batch
        male_dis_arr, female_dis_arr = [], []
        for i in range(10):
            male_distr = torch.histc(male_batch[:, i], bins=800, min=min_tensor[i].item(), max=max_tensor[i].item()) \
                        / male_index.shape[0]
            female_distr = torch.histc(female_batch[:, i], bins=800, min=min_tensor[i].item(), max=max_tensor[i].item()) \
                        / female_index.shape[0]
            male_dis_arr.append(male_distr.detach().numpy())
            female_dis_arr.append(female_distr.detach().numpy())
            # JS distance calculation
        KLD = KL_divergence(np.array(male_dis_arr), np.array(female_dis_arr))

        mse_Loss = age_loss + detail_in_re_loss + detail_oc_re_loss + wage_loss + cap_gain_loss + cap_loss_loss + \
                   div_stock_loss + num_emp_loss + own_bus_loss + vet_ben_loss + wek_yea_loss

        ce_Loss = cls_wor_loss + edu_loss + enr_wk_loss + mar_sta_loss + maj_ind_loss + maj_occ_loss + race_loss + \
                  his_ori_loss + sex_loss + mem_lab_loss + rea_une_loss + ful_par_loss + tax_loss + reg_res_loss + \
                  fam_sta_loss + sum_hou_loss + fam_u18_loss

        # different alpha 0.1 ~ 1
        alpha = par.RATIO_JSD
        Multi_loss = (1 - alpha)*(mse_Loss + ce_Loss) + alpha * KLD

        loss_item.append(mse_Loss)
        loss_item.append(ce_Loss)
        loss_item.append(alpha * KLD)

        return Multi_loss, loss_item


# ============================== classifier loss function ====================================
class Loss_fairness_regularization(nn.Module):
    def __init__(self):
        super(Loss_fairness_regularization, self).__init__()

    def forward(self, label_pred, label_true):
        loss_item = []
        # ======binary cross entropy======
        target = label_true[:][:, 0].clone().detach().view(len(label_true), 1)
        CE_loss = F.binary_cross_entropy(label_pred, target)

        # ======equalized odds calculation======
        male_tp, male_tn = 0, 0
        female_tp, female_tn = 0, 0
        male_p, male_n = 0, 0
        female_p, female_n = 0, 0
        female = torch.tensor([1, 0]).type(torch.float32)
        male = torch.tensor([0, 1]).type(torch.float32)
        for i in range(len(label_pred)):
            gender = label_true[i][1:3]
            if torch.equal(gender, female) is True:
                if label_pred[i].detach().numpy() >= par.SIG_THRESHOLD:
                    pred = 1
                    female_p += 1
                    if pred == label_true[i][0].detach().numpy():
                        female_tp += 1
                else:
                    pred = 0
                    female_n += 1
                    if pred == label_true[i][0].detach().numpy():
                        female_tn += 1
            elif torch.equal(gender, male) is True:
                if label_pred[i].detach().numpy() >= par.SIG_THRESHOLD:
                    pred = 1
                    male_p += 1
                    if pred == label_true[i][0].detach().numpy():
                        male_tp += 1
                else:
                    pred = 0
                    male_n += 1
                    if pred == label_true[i][0].detach().numpy():
                        male_tn += 1
        # confusion matrix
        tp = {'female': female_tp, 'male': male_tp}
        tn = {'female': female_tn, 'male': male_tn}
        fp = {'female': female_p - female_tp, 'male': male_p - male_tp}
        fn = {'female': female_n - female_tn, 'male': male_n - male_tn}
        Attr = ['female', 'male']

        # regularization item
        EO_regulization = Equalized_odds(tp, tn, fn, fp, Attr)

        beta = par.RATIO_EO
        Loss_fair = CE_loss + beta * EO_regulization
        loss_item.append(CE_loss)
        loss_item.append(EO_regulization)

        return Loss_fair, loss_item

