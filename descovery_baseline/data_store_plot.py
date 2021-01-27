import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


from calculate import data_processsing
from calculate import accuracy_fairness
from calculate import models_define

from sklearn.model_selection import train_test_split

data_sheet = []
threshold_list = []
accuracy_array = np.zeros((6, 9), dtype=np.float64)
SP_array = np.zeros((6, 9), dtype=np.float64)
PP_array = np.zeros((6, 9), dtype=np.float64)
PE_array = np.zeros((6, 9), dtype=np.float64)
EOp_array = np.zeros((6, 9), dtype=np.float64)
EO_array = np.zeros((6, 9), dtype=np.float64)
CAE_array = np.zeros((6, 9), dtype=np.float64)
OAE_array = np.zeros((6, 9), dtype=np.float64)
TE_array = np.zeros((6, 9), dtype=np.float64)


dataset_adult_before = pd.read_csv("./doc/adult_census_income/adult.csv")
dataset_adult = data_processsing(dataset_adult_before)

'''original data'''
# data split
array = dataset_adult.values
X = array[:, 0:-1]
Y = array[:, -1]
# 20% of total data as validation
validation_size = 0.20
seed = 7
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
models = models_define()

# fairness
id_female = np.where(X_validation[:, 8] == 1)[0]
id_male = np.where(X_validation[:, 8] == 0)[0]
t = 0 # initialize t
for n in range(9):
    t = t + 0.1
    threshold_list.append(t)
    data = accuracy_fairness(models, id_female, id_male, X_train, Y_train, X_validation, Y_validation, t)
    for m in range(6):
        accuracy_array[m][n] = data[m][0]
        SP_array[m][n] = data[m][1]
        PP_array[m][n] = data[m][2]
        PE_array[m][n] = data[m][3]
        EOp_array[m][n] = data[m][4]
        EO_array[m][n] = data[m][5]
        CAE_array[m][n] = data[m][6]
        OAE_array[m][n] = data[m][7]
        TE_array[m][n] = data[m][8]
    data_sheet.append(data)

df_acc = DataFrame(data=accuracy_array, index=['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF'], columns=threshold_list)
df_SP = DataFrame(data=SP_array, index=['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF'], columns=threshold_list)
df_PP = DataFrame(data=PP_array, index=['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF'], columns=threshold_list)
df_PE = DataFrame(data=PE_array, index=['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF'], columns=threshold_list)
df_EOp = DataFrame(data=EOp_array, index=['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF'], columns=threshold_list)
df_EO = DataFrame(data=EO_array, index=['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF'], columns=threshold_list)
df_CAE = DataFrame(data=CAE_array, index=['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF'], columns=threshold_list)
df_OAE = DataFrame(data=OAE_array, index=['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF'], columns=threshold_list)
df_TE = DataFrame(data=TE_array, index=['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF'], columns=threshold_list)

# save in excel
writer = pd.ExcelWriter('original.xlsx')
df_acc.to_excel(writer, 'accuracy')
df_SP.to_excel(writer, 'Statistical Parity')
df_PP.to_excel(writer, 'Predictive Parity')
df_PE.to_excel(writer, 'Predictive Equality')
df_EOp.to_excel(writer, 'Equal Opportunity')
df_EO.to_excel(writer, 'Equalized Odds')
df_CAE.to_excel(writer, 'Conditional use accuracy equality')
df_OAE.to_excel(writer, 'Overall accuracy equality')
df_TE.to_excel(writer, 'Treatment equality')
writer.save()


color_list = ['r', 'orange', 'lawngreen', 'blueviolet', 'fuchsia', 'dodgerblue']
label_list = ['LR', 'LDA', 'KNN', 'CART', 'NB', 'RF']

'''accuracy-threshold plot'''
fig = plt.figure()
for i in range(6):
    plt.plot(threshold_list, df_acc.iloc[i], c=color_list[i], alpha=0.7, label=label_list[i])
plt.legend()
plt.xlabel('threshold', fontsize=10)
plt.ylabel('accuracy(%)', fontsize=10)
plt.title('accuracy-threshold curve', fontsize=12)
plt.show()

''' fairness-threshold plot'''
# SP
fig = plt.figure()
for i in range(6):
    plt.plot(threshold_list, df_SP.iloc[i], c=color_list[i], alpha=0.7, label=label_list[i])
plt.legend()
plt.xlabel('threshold', fontsize=10)
plt.ylabel('Statistical Parity', fontsize=10)
plt.title('SP-threshold curve', fontsize=12)
plt.show()

# PP
fig = plt.figure()
for i in range(6):
    plt.plot(threshold_list, df_PP.iloc[i], c=color_list[i], alpha=0.7, label=label_list[i])
plt.legend()
plt.xlabel('threshold', fontsize=10)
plt.ylabel('Predictive Parity', fontsize=10)
plt.title('PP-threshold curve', fontsize=12)
plt.show()

# PE
fig = plt.figure()
for i in range(6):
    plt.plot(threshold_list, df_PE.iloc[i], c=color_list[i], alpha=0.7, label=label_list[i])
plt.legend()
plt.xlabel('threshold', fontsize=10)
plt.ylabel('Predictive Equality', fontsize=10)
plt.title('PE-threshold curve', fontsize=12)
plt.show()

# EOp
fig = plt.figure()
for i in range(6):
    plt.plot(threshold_list, df_EOp.iloc[i], c=color_list[i], alpha=0.7, label=label_list[i])
plt.legend()
plt.xlabel('threshold', fontsize=10)
plt.ylabel('Equal Opportunity', fontsize=10)
plt.title('EOp-threshold curve', fontsize=12)
plt.show()

# EO
fig = plt.figure()
for i in range(6):
    plt.plot(threshold_list, df_EO.iloc[i], c=color_list[i], alpha=0.7, label=label_list[i])
plt.legend()
plt.xlabel('threshold', fontsize=10)
plt.ylabel('Equalized Odds', fontsize=10)
plt.title('EO-threshold curve', fontsize=12)
plt.show()

# CAE
fig = plt.figure()
for i in range(6):
    plt.plot(threshold_list, df_CAE.iloc[i], c=color_list[i], alpha=0.7, label=label_list[i])
plt.legend()
plt.xlabel('threshold', fontsize=10)
plt.ylabel('Conditional use accuracy equality', fontsize=10)
plt.title('CAE-threshold curve', fontsize=12)
plt.show()

# OAE
fig = plt.figure()
for i in range(6):
    plt.plot(threshold_list, df_OAE.iloc[i], c=color_list[i], alpha=0.7, label=label_list[i])
plt.legend()
plt.xlabel('threshold', fontsize=10)
plt.ylabel('Overall accuracy equality', fontsize=10)
plt.title('OAE-threshold curve', fontsize=12)
plt.show()

# TE
fig = plt.figure()
for i in range(6):
    plt.plot(threshold_list, df_TE.iloc[i], c=color_list[i], alpha=0.7, label=label_list[i])
plt.legend()
plt.xlabel('threshold', fontsize=10)
plt.ylabel('Treatment equality', fontsize=10)
plt.title('TE-threshold curve', fontsize=12)
plt.show()


'''
# drop sex attribute
dataset_drop_sex = dataset_adult.copy(deep=True)
dataset_drop_sex.drop(labels='sex', axis=1, inplace=True)

# data split
array = dataset_drop_sex.values
X = array[:,0:-1]
Y = array[:,-1]
# 20% of total data as validation
validation_size = 0.20
seed = 7
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
models_sex = models_define()

# fairness
# print("Results dropping sex")
data_sex = accuracy_fairness(models_sex, id_female, id_male, X_train, Y_train, X_validation, Y_validation)

# drop sex and relationship
dataset_drop_sex_rel = dataset_adult.copy(deep=True)
dataset_drop_sex_rel.drop(labels='sex', axis=1, inplace=True)
# drop relationship
dataset_drop_sex_rel.drop(labels='relationship', axis=1, inplace=True)

# data split
array = dataset_drop_sex_rel.values
X = array[:,0:-1]
Y = array[:,-1]
# 20% of total data as validation
validation_size = 0.20
seed = 7
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
models_sex_rel = models_define()

# fairness
# print("Results dropping sex and relationship")
data_sex_rel = accuracy_fairness(models_sex_rel, id_female, id_male, X_train, Y_train, X_validation, Y_validation)

# drop sex, relationship and marital.status
dataset_drop_sex_rel_mar = dataset_adult.copy(deep=True)
dataset_drop_sex_rel_mar.drop(labels='sex', axis=1, inplace=True)
# drop relationship
dataset_drop_sex_rel_mar.drop(labels='relationship', axis=1, inplace=True)
# drop marital.status
dataset_drop_sex_rel_mar.drop(labels='marital.status', axis=1, inplace=True)

# data split
array = dataset_drop_sex_rel_mar.values
X = array[:,0:-1]
Y = array[:,-1]
# 20% of total data as validation
validation_size = 0.20
seed = 7
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
models_sex_rel_mar = models_define()

# fairness
# print("Results dropping sex, relationship and marital status")
data_sex_rel_mar = accuracy_fairness(models_sex_rel_mar, id_female, id_male, X_train, Y_train, X_validation, Y_validation)

# reconstruct dataset pre classifier
data_LR, data_LDA, data_KNN, data_CART, data_NB, data_RF = [], [], [], [], [], []
data_sheet = []

'''
