import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def feature_string2scalar(string_feature):
    scalar_feature =string_feature.copy(deep=True)
    unique_labels = string_feature.unique()
    for index, label in enumerate(unique_labels):
        scalar_feature[string_feature==label] = index
    return scalar_feature.astype(int)

def models_define(num_trees = 100,max_features = 3):
    #Spot Check 5 Algorithms (LR, LDA, KNN, CART, GNB, SVM)
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))
    # models.append(('SVM', SVC(kernel='rbf')))
    return models

def calculate_fairness(id_f, id_m, Y_val, predictions):
    accuracy_f = 100 * accuracy_score(Y_val[id_f], predictions[id_f])
    accuracy_m = 100 * accuracy_score(Y_val[id_m], predictions[id_m])
    # statistical parity
    SP_f = 100 * np.where(predictions[id_f] == 1)[0].size / id_f.size
    SP_m = 100 * np.where(predictions[id_m] == 1)[0].size / id_m.size
    SP = np.abs(SP_f - SP_m)
    # tp,tn,fp,fn female
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(Y_val[id_f], predictions[id_f]).ravel()
    tpr_f = 100 * tp_f / (tp_f + fn_f)
    fpr_f = 100 * fp_f / (fp_f + tn_f)
    ppv_f = 100 * tp_f / (tp_f + fp_f)
    fnr_f = 100 * fn_f / (tp_f + fn_f)
    npv_f = 100 * tn_f / (tn_f + fn_f)
    tnr_f = 100 * tn_f / (tn_f + fp_f)
    # tp,tn,fp,fn male
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(Y_val[id_m], predictions[id_m]).ravel()
    tpr_m = 100 * tp_m / (tp_m + fn_m)
    fpr_m = 100 * fp_m / (fp_m + tn_m)
    ppv_m = 100 * tp_m / (tp_m + fp_m)
    fnr_m = 100 * fn_m / (tp_m + fn_m)
    npv_m = 100 * tn_m / (tn_m + fn_m)
    tnr_m = 100 * tn_m / (tn_m + fp_m)
    # predictive parity
    PP = np.abs(ppv_f - ppv_m)
    # predictive equality
    PE = np.abs(fpr_f - fpr_m)
    # equal opportunity
    EOp = np.abs(fnr_f - fnr_m)
    # equalized odds
    EO = np.abs(tpr_f - tpr_m) + np.abs(tnr_f - tnr_m)
    # conditional use accuracy equality
    CAE = np.abs(ppv_f - ppv_m) + np.abs(npv_f - npv_m)
    # overall accuracy equality
    OAE = np.abs(accuracy_f - accuracy_m)
    # Treatment equality
    TE = np.abs(fn_f / fp_f - fn_m / fp_m)

    return SP, PP, PE, EOp, EO, CAE, OAE, TE

def accuracy_fairness(models, id_f, id_m, X_train, Y_train, X_val, Y_val, threshold):
    Acc, SP, PP, PE, EOp, EO, CAE, OAE, TE = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for name, model in models:
        model.fit(X_train, Y_train)
        # TODO
        # predictions = model.predict(X_val) # default threshold is 0.5
        if threshold == None:
            threshold = 0.5

        predictions = model.predict_proba(X_val)[:, 1] > threshold  # modify threshold to 0.05
        # print(predictions)

        Acc[name] = 100 * accuracy_score(Y_val, predictions)
        SP[name], PP[name], PE[name], EOp[name], EO[name], CAE[name], OAE[name], TE[name] = calculate_fairness(
            id_f, id_m, Y_val, predictions)
        msg = "%s: %f, %f, %f, %f, %f, %f, %f, %f, %f" % (name, Acc[name], SP[name], PP[name], PE[name],
                                                          EOp[name], EO[name], CAE[name], OAE[name], TE[name])
        # print(msg)
        # convert list to numpy
    Acc, SP, PP, PE, EOp, EO, CAE, OAE, TE = np.array(list(Acc.values()))[:, None], np.array(list(SP.values()))[:, None], \
                                             np.array(list(PP.values()))[:, None], np.array(list(PE.values()))[:, None], \
                                             np.array(list(EOp.values()))[:, None], np.array(list(EO.values()))[:, None], \
                                             np.array(list(CAE.values()))[:, None], np.array(list(OAE.values()))[:, None], \
                                             np.array(list(TE.values()))[:, None]
    data = np.concatenate((Acc, SP, PP, PE, EOp, EO, CAE, OAE, TE), axis=1)
    # print(data)
    return data

def data_processsing(dataset_adult_before):
    dataset_adult = dataset_adult_before
    # data processing
    dataset_adult['education'] = feature_string2scalar(dataset_adult['education'] )
    dataset_adult['native.country'] = feature_string2scalar(dataset_adult['native.country'] )
    dataset_adult['relationship'] = feature_string2scalar(dataset_adult['relationship'] )
    dataset_adult['workclass'] = feature_string2scalar(dataset_adult['workclass'] )
    dataset_adult['occupation'] = feature_string2scalar(dataset_adult['occupation'] )
    dataset_adult['marital.status'] = feature_string2scalar(dataset_adult['marital.status'] )
    dataset_adult.drop(labels='fnlwgt', axis = 1, inplace = True)

    # Convert income value to 0 and 1
    dataset_adult['income'] = dataset_adult['income'].map({'<=50K': 0, '>50K': 1})
    # Convert Sex value to 0 and 1
    dataset_adult["sex"] = dataset_adult["sex"].map({"Male": 0, "Female": 1})
    # Convert race value to 0 and 1
    dataset_adult["race"] = feature_string2scalar(dataset_adult["race"])
    return dataset_adult


