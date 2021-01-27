import math
import torch
import torch.utils.data as Data
import pandas as pd
from functions.utils import feature_nominal2scalar, numeric_tensor, one_hot_tensor

# ============== original dataset maker ====================
def Dataset_maker(URL, mode=''):
    """
    read csv data and make the dataset
    """
    dataset_adult = pd.read_csv(URL)
    data_test = dataset_adult
    """
    transform data to tensor
    """
    # ========= map income to scalar value =========
    data_test['income'] = data_test['income'].map({'<=50K': 0, '>50K': 1})
    # ========= transform nominal attributes to scalar =========
    dataset_test_copy = data_test.copy(deep=True)
    dataset_test_copy['class of worker'] = feature_nominal2scalar(data_test['class of worker'])
    dataset_test_copy['education'] = feature_nominal2scalar(data_test['education'])
    dataset_test_copy['enroll in edu inst last wk'] = feature_nominal2scalar(data_test['enroll in edu inst last wk'])
    dataset_test_copy['marital stat'] = feature_nominal2scalar(data_test['marital stat'])
    dataset_test_copy['major industry code'] = feature_nominal2scalar(data_test['major industry code'])
    dataset_test_copy['major occupation code'] = feature_nominal2scalar(data_test['major occupation code'])
    dataset_test_copy['race'] = feature_nominal2scalar(data_test['race'])
    dataset_test_copy['hispanic origin'] = feature_nominal2scalar(data_test['hispanic origin'])
    dataset_test_copy['sex'] = feature_nominal2scalar(data_test['sex'])
    dataset_test_copy['member of a labor union'] = feature_nominal2scalar(data_test['member of a labor union'])
    dataset_test_copy['reason for unemployment'] = feature_nominal2scalar(data_test['reason for unemployment'])
    dataset_test_copy['full or part time employment stat'] = feature_nominal2scalar(data_test['full or part time employment stat'])
    dataset_test_copy['tax filer stat'] = feature_nominal2scalar(data_test['tax filer stat'])
    dataset_test_copy['region of previous residence'] = feature_nominal2scalar(data_test['region of previous residence'])
    dataset_test_copy['detailed household and family stat'] = feature_nominal2scalar(data_test['detailed household and family stat'])
    dataset_test_copy['detailed household summary in household'] = feature_nominal2scalar(data_test['detailed household summary in household'])
    dataset_test_copy['family members under 18'] = feature_nominal2scalar(data_test['family members under 18'])
    # ======== transform numerical attributes to tensor ==========
    age_numeric = numeric_tensor(dataset_test_copy['age'])
    detail_in_re_numeric = numeric_tensor(dataset_test_copy['detailed industry recode'])
    detail_oc_re_numeric = numeric_tensor(dataset_test_copy['detailed occupation recode'])
    wage_numeric = numeric_tensor(dataset_test_copy['wage per hour'])
    cap_gain_numeric = numeric_tensor(dataset_test_copy['capital gains'])
    cap_loss_numeric = numeric_tensor(dataset_test_copy['capital losses'])
    div_stock_numeric = numeric_tensor(dataset_test_copy['dividends from stocks'])
    num_emp_numeric = numeric_tensor(dataset_test_copy['num persons worked for employer'])
    own_bus_numeric = numeric_tensor(dataset_test_copy['own business or self employed'])
    vet_ben_numeric = numeric_tensor(dataset_test_copy['veterans benefits'])
    wek_yea_numeric = numeric_tensor(dataset_test_copy['weeks worked in year'])
    # ======== transform scalar attributes to one-hot code =========
    cls_wor_onehot = one_hot_tensor(dataset_test_copy['class of worker'])
    edu_onehot = one_hot_tensor(dataset_test_copy['education'])
    enr_wk_onehot = one_hot_tensor(dataset_test_copy['enroll in edu inst last wk'])
    mar_sta_onehot = one_hot_tensor(dataset_test_copy['marital stat'])
    maj_ind_onehot = one_hot_tensor(dataset_test_copy['major industry code'])
    maj_occ_onehot = one_hot_tensor(dataset_test_copy['major occupation code'])
    race_onehot = one_hot_tensor(dataset_test_copy['race'])
    his_ori_onehot = one_hot_tensor(dataset_test_copy['hispanic origin'])
    sex_onehot = one_hot_tensor(dataset_test_copy['sex'])
    mem_lab_onehot = one_hot_tensor(dataset_test_copy['member of a labor union'])
    rea_une_onehot = one_hot_tensor(dataset_test_copy['reason for unemployment'])
    ful_par_onehot = one_hot_tensor(dataset_test_copy['full or part time employment stat'])
    tax_sta_onehot = one_hot_tensor(dataset_test_copy['tax filer stat'])
    reg_res_onehot = one_hot_tensor(dataset_test_copy['region of previous residence'])
    fam_sta_onehot = one_hot_tensor(dataset_test_copy['detailed household and family stat'])
    sum_hou_onehot = one_hot_tensor(dataset_test_copy['detailed household summary in household'])
    fam_u18_onehot = one_hot_tensor(dataset_test_copy['family members under 18'])

    income_label = numeric_tensor(dataset_test_copy['income'])

    # ======= combine all attribute to dataset =========
    dataset_kdd = torch.cat((age_numeric, cls_wor_onehot, detail_in_re_numeric, detail_oc_re_numeric, edu_onehot,
                             wage_numeric, enr_wk_onehot, mar_sta_onehot, maj_ind_onehot, maj_occ_onehot, race_onehot,
                             his_ori_onehot, sex_onehot, mem_lab_onehot, rea_une_onehot, ful_par_onehot,
                             cap_gain_numeric, cap_loss_numeric, div_stock_numeric, tax_sta_onehot, reg_res_onehot,
                             fam_sta_onehot, sum_hou_onehot, num_emp_numeric, fam_u18_onehot, own_bus_numeric,
                             vet_ben_numeric, wek_yea_numeric), 1)
    print(dataset_kdd.size())
    if mode == '':
        Mydataset = Data.TensorDataset(dataset_kdd, income_label)
    elif mode == 'c':
        # construct a combine-label
        combine_label = torch.cat((income_label, sex_onehot), 1)
        Mydataset = Data.TensorDataset(dataset_kdd, combine_label)

    return Mydataset


# ======================= classification dataset maker ===============================
def Dataset_maker_clf_distribution(URL, proportion):
    """
    URL: dataset,
    proportion (list): [train, val, test]
    """
    quantity = []
    dataset_list = []
    dataset_tensor_list = []
    female_set_num, male_set_num = [], []

    data_test = pd.read_csv(URL)
    # seperate dataset in two gender
    data_female = data_test.loc[data_test['sex'] == 'Female']
    data_male = data_test.loc[data_test['sex'] == 'Male']
    data_female.reset_index(drop=True, inplace=True)
    data_male.reset_index(drop=True, inplace=True)
    # regenerate train, val, test with equal gender distribution
    quantity.append(len(data_female))
    quantity.append(len(data_male))
    for i in range(2):
        train_num = quantity[i] - math.ceil(quantity[i] * proportion[1]) - math.ceil(quantity[i] * proportion[2])
        val_num = math.ceil(quantity[i] * proportion[1])
        test_num = math.ceil(quantity[i] * proportion[2])
        if i == 0:
            female_set_num.append(train_num)
            female_set_num.append(val_num)
            female_set_num.append(test_num)
        elif i == 1:
            male_set_num.append(train_num)
            male_set_num.append(val_num)
            male_set_num.append(test_num)

    # step 2: shuffle female and male dataset respectively
    data_female = data_female.sample(frac=1.0)
    data_female.reset_index(drop=True, inplace=True)
    train_female = data_female.loc[0: female_set_num[0] - 1]
    val_female = data_female.loc[female_set_num[0]: sum(female_set_num[0:2]) - 1]
    test_female = data_female.loc[sum(female_set_num[0:2]): sum(female_set_num)]

    data_male = data_male.sample(frac=1.0)
    data_male.reset_index(drop=True, inplace=True)
    train_male = data_male.loc[0: male_set_num[0] - 1]
    val_male = data_male.loc[male_set_num[0]: sum(male_set_num[0:2]) - 1]
    test_male = data_male.loc[sum(male_set_num[0:2]): sum(male_set_num)]

    # step 3: combine female and male data set together
    train_data = pd.concat([train_female, train_male])
    val_data = pd.concat([val_female, val_male])
    test_data = pd.concat([test_female, test_male])

    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    # step 4: shuffle val dataset and test dataset (ensure that the first line of val and test is female)
    # because in clf training process the train set will be shuffled so we do not shuffle train set here
    val_head = val_data.loc[0:0]
    val_shuffle = val_data.loc[1:len(val_data)].sample(frac=1.0)
    val_data = pd.concat([val_head, val_shuffle])
    val_data.reset_index(drop=True, inplace=True)
    test_head = test_data.loc[0:0]
    test_shuffle = test_data.loc[1:len(test_data)].sample(frac=1.0)
    test_data = pd.concat([test_head, test_shuffle])
    test_data.reset_index(drop=True, inplace=True)

    dataset_list.append(train_data)
    dataset_list.append(val_data)
    dataset_list.append(test_data)

    # transform dataframe to tensor
    for j in range(2):
        dataset_list[j]['income'] = dataset_list[j]['income'].map({'<=50K': 0, '>50K': 1})
        dataset_list[j]['sex'] = feature_nominal2scalar(dataset_list[j]['sex'])
        sex_one_hot = one_hot_tensor(dataset_list[j]['sex'])

        # ======transform numerical attribute to tensor ======
        f1_numeric = numeric_tensor(dataset_list[j]['feature_1'])
        f2_numeric = numeric_tensor(dataset_list[j]['feature_2'])
        f3_numeric = numeric_tensor(dataset_list[j]['feature_3'])
        f4_numeric = numeric_tensor(dataset_list[j]['feature_4'])
        f5_numeric = numeric_tensor(dataset_list[j]['feature_5'])
        f6_numeric = numeric_tensor(dataset_list[j]['feature_6'])
        f7_numeric = numeric_tensor(dataset_list[j]['feature_7'])
        f8_numeric = numeric_tensor(dataset_list[j]['feature_8'])
        f9_numeric = numeric_tensor(dataset_list[j]['feature_9'])
        f10_numeric = numeric_tensor(dataset_list[j]['feature_10'])
        income_label = numeric_tensor(dataset_list[j]['income'])

        dataset_encoded = torch.cat((f1_numeric, f2_numeric, f3_numeric, f4_numeric, f5_numeric, f6_numeric, f7_numeric,
                                     f8_numeric, f9_numeric, f10_numeric), 1)
        # construct a combine-label
        combine_label = torch.cat((income_label, sex_one_hot), 1)
        Mydataset = Data.TensorDataset(dataset_encoded, combine_label)

        dataset_tensor_list.append(Mydataset)

    return dataset_tensor_list[0], dataset_tensor_list[1], test_data


def Dataset_maker_classification(URL):
    encoded_adult_dataset_test = pd.read_csv(URL)
    print(encoded_adult_dataset_test.shape)
    encoded_adult_dataset_test['income'] = encoded_adult_dataset_test['income'].map({'<=50K': 0, '>50K': 1})
    encoded_adult_dataset_test['sex'] = feature_nominal2scalar(encoded_adult_dataset_test['sex'])
    sex_one_hot = one_hot_tensor(encoded_adult_dataset_test['sex'])
    # ======transform numerical attribute to tensor ======
    f1_numeric = numeric_tensor(encoded_adult_dataset_test['feature_1'])
    f2_numeric = numeric_tensor(encoded_adult_dataset_test['feature_2'])
    f3_numeric = numeric_tensor(encoded_adult_dataset_test['feature_3'])
    f4_numeric = numeric_tensor(encoded_adult_dataset_test['feature_4'])
    f5_numeric = numeric_tensor(encoded_adult_dataset_test['feature_5'])
    f6_numeric = numeric_tensor(encoded_adult_dataset_test['feature_6'])
    f7_numeric = numeric_tensor(encoded_adult_dataset_test['feature_7'])
    f8_numeric = numeric_tensor(encoded_adult_dataset_test['feature_8'])
    f9_numeric = numeric_tensor(encoded_adult_dataset_test['feature_9'])
    f10_numeric = numeric_tensor(encoded_adult_dataset_test['feature_10'])
    income_label = numeric_tensor(encoded_adult_dataset_test['income'])
    # =======combine all attributes to dataset =======
    dataset_encoded = torch.cat((f1_numeric, f2_numeric, f3_numeric, f4_numeric, f5_numeric, f6_numeric, f7_numeric,
                                 f8_numeric, f9_numeric, f10_numeric), 1)

    # construct a combine-label
    combine_label = torch.cat((income_label, sex_one_hot), 1)
    Mydataset = Data.TensorDataset(dataset_encoded, combine_label)

    return Mydataset