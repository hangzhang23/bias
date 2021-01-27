import torch
import numpy as np
import parameters as par
from torch.autograd import Variable

# =======================
# we set the threshold to 0.5, majority is '<=50K'(0), minority is '>=50K'(1)
#
# =======================


def accuracy_calculate(data_loader, model):
    correct = 0
    total = 0
    for i, (data, label) in enumerate(data_loader):
        data = Variable(data)
        label = Variable(label).type(torch.float32)
        output = model(data)
        if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
            pred = 1
            total += 1
            if pred == label.detach().numpy()[0]:
                correct += 1

        else:
            pred = 0
            total += 1
            if pred == label.detach().numpy()[0]:
                correct += 1
    acc = correct / total
    print("correct/total:(%s/%s)" % (correct, total))
    print("accuracy=", acc)


def balanced_accuracy(dataloader, model):
    correct = 0
    total = 0
    tp = 0
    tn = 0
    P = 0
    N = 0
    for i, (data, label) in enumerate(dataloader):
        data = Variable(data)
        label = Variable(label).type(torch.float32)
        output = model(data)
        if output.detach.numpy()[0] >= par.SIG_THRESHOLD:
            pred = 1
            total += 1
            P += 1
            if pred == label.detach().numpy()[0]:
                correct += 1
                tp += 1
        else:
            pred = 0
            total += 1
            N += 1
            if pred == label.detach().numpy()[0]:
                correct += 1
                tn += 1
    acc = (tp + tn) / (P + N)
    print("correct/total:(%s/%s)" % (correct, total))
    print("tp + tn:", (tp + tn))
    print("p + n:", (P + N))
    print("acc:", acc)


def accuracy_calculate_gender(dataset, model):
    # male
    male_correct = 0
    female_correct = 0
    male_total = 0
    female_total = 0
    female = torch.tensor([1, 0]).type(torch.float32)
    male = torch.tensor([0, 1]).type(torch.float32)

    for i, (data, label) in enumerate(dataset):
        data = Variable(data)
        gt = Variable(label)
        label = torch.tensor([gt[0]])
        gender = gt[1:3]
        output = model(data)
        if torch.equal(gender, female) is True:
            female_total += 1
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                if pred == label.detach().numpy()[0]:
                    female_correct += 1
            else:
                pred = 0
                if pred == label.detach().numpy()[0]:
                    female_correct += 1
        else:
            male_total += 1
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                if pred == label.detach().numpy()[0]:
                    male_correct += 1
            else:
                pred = 0
                if pred == label.detach().numpy()[0]:
                    male_correct += 1

    male_acc = male_correct / male_total
    female_acc = female_correct / female_total
    total_acc = (male_correct + female_correct) / (male_total + female_total)
    # print("female income predict accuracy:%s" % female_acc)
    # print("male income predict accuracy:%s" % male_acc)
    # print("total predict accuracy:%s" % ((male_correct+female_correct)/total))
    return total_acc, male_acc, female_acc


def balanced_accuracy_gender(dataset, model):
    male_tp, male_tn = 0, 0
    female_tp, female_tn = 0, 0
    male_p, male_n = 0, 0
    female_p, female_n = 0, 0
    female = torch.tensor([1, 0]).type(torch.float32)
    male = torch.tensor([0, 1]).type(torch.float32)

    for i, (data, label) in enumerate(dataset):
        data = Variable(data)
        gt = Variable(label)
        label = torch.tensor([gt[0]])
        gender = gt[1:3]
        output = model(data)
        if torch.equal(gender, female) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                female_p += 1
                if pred == label.detach().numpy()[0]:
                    female_tp += 1
            else:
                pred = 0
                female_n += 1
                if pred == label.detach().numpy()[0]:
                    female_tn += 1
        elif torch.equal(gender, male) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                male_p += 1
                if pred == label.detach().numpy()[0]:
                    male_tp += 1
            else:
                pred = 0
                male_n += 1
                if pred == label.detach().numpy()[0]:
                    male_tn += 1

    # confusion matrix
    tp = {'female': female_tp, 'male': male_tp}
    tn = {'female': female_tn, 'male': male_tn}
    fp = {'female': female_p - female_tp, 'male': male_p - male_tp}
    fn = {'female': female_n - female_tn, 'male': male_n - male_tn}
    Attr = ['female', 'male']
    # print("tp:", tp)
    # print("tn:", tn)
    # print("fp:", fp)
    # print("fn:", fn)

    # normal accuracy
    female_acc = (female_tp + female_tn) / (female_p + female_n)
    male_acc = (male_tp + male_tn) / (male_p + male_n)
    total_acc = (female_tp + female_tn + male_tp + male_tn) / (female_p + female_n + male_p + male_n)
    acc = {'female': female_acc, 'male': male_acc}

    # balanced accuracy: 0.5*(TP/P + TN/N)
    balanced_female_acc = 0.5 * (
                (female_tp / (female_n - female_tn + female_tp)) + (female_tn / (female_p - female_tp + female_tn)))
    balanced_male_acc = 0.5 * ((male_tp / (male_n - male_tn + male_tp)) + (male_tn / (male_p - male_tp + male_tn)))
    balanced_total_acc = 0.5 * (
                ((female_tp + male_tp) / (female_n - female_tn + female_tp + male_n - male_tn + male_tp))
                + ((female_tn + male_tn) / (female_p - female_tp + female_tn + male_p - male_tp + male_tn)))

    acc_b = {'female': balanced_female_acc, 'male': balanced_male_acc}

    print("female income predict accuracy/balanced female income predict accuracy:(%s/%s)" % (
    female_acc, balanced_female_acc))
    print("male income predict accuracy/balanced male income predict accuracy:(%s/%s)" % (male_acc, balanced_male_acc))
    print("total income predict accuracy/balanced total income predict accuracy:(%s/%s)" % (
    total_acc, balanced_total_acc))
    print("SP:", np.abs((female_p / (female_p + female_n)) - (male_p / (male_p + male_n))))

    # fairness calculate
    fairness_calculate(tp, tn, fn, fp, acc, acc_b, Attr)


def fairness_calculate(tp, tn, fn, fp, acc, acc_b, A):
    att_category = A
    tpr, fpr, ppv, fnr, npv, tnr = {}, {}, {}, {}, {}, {}
    # female\male fairness
    for a in att_category:
        tpr[a] = 100 * tp[a] / (tp[a] + fn[a])
        fpr[a] = 100 * fp[a] / (fp[a] + tn[a])
        ppv[a] = 100 * tp[a] / (tp[a] + fp[a])
        fnr[a] = 100 * fn[a] / (tp[a] + fn[a])
        npv[a] = 100 * tn[a] / (tn[a] + fn[a])
        tnr[a] = 100 * tn[a] / (tn[a] + fp[a])

    # predictive parity
    PP = np.abs(ppv[att_category[0]] - ppv[att_category[1]])
    # predictive equality
    PE = np.abs(fpr[att_category[0]] - fpr[att_category[1]])
    # equal opportunity
    EOp = np.abs(fnr[att_category[0]] - fnr[att_category[1]])
    # equalized odds
    EO = np.abs(tpr[att_category[0]] - tpr[att_category[1]]) + np.abs(fpr[att_category[0]] - fpr[att_category[1]])
    # conditional use accuracy equality
    CAE = np.abs(ppv[att_category[0]] - ppv[att_category[1]]) + np.abs(npv[att_category[0]] - npv[att_category[1]])
    # overall accuracy equality
    OAE = np.abs(acc[att_category[0]] - acc[att_category[1]])
    # overall accuracy equality balanced
    OAE_b = np.abs(acc_b[att_category[0]] - acc_b[att_category[1]])
    # treatment equality
    TE = np.abs(fn[att_category[0]] / fp[att_category[0]] - fn[att_category[1]] / fp[att_category[1]])

    print("PP:%s, PE:%s, EOp:%s, EO:%s, CAE:%s, TE:%s" % (PP, PE, EOp, EO, CAE, TE))
    print("OAE/balanced OAE:%s/%s" % (OAE, OAE_b))


def Equalized_odds(tp, tn, fn, fp, A):
    att_category = A
    tpr, fpr, ppv, fnr, npv, tnr = {}, {}, {}, {}, {}, {}
    # fairness items calculation
    for a in att_category:
        tpr[a] = 100 * tp[a] / (tp[a] + fn[a])
        fpr[a] = 100 * fp[a] / (fp[a] + tn[a])
        # ppv[a] = 100 * tp[a] / (tp[a] + fp[a])
        # fnr[a] = 100 * fn[a] / (tp[a] + fn[a])
        # npv[a] = 100 * tn[a] / (tn[a] + fn[a])
        # tnr[a] = 100 * tn[a] / (tn[a] + fp[a])

    # equalized odds
    EO = np.abs(tpr[att_category[0]] - tpr[att_category[1]]) + np.abs(fpr[att_category[0]] - fpr[att_category[1]])

    return EO


def EO_evaluation(dataset, model):
    male_tp, male_tn = 0, 0
    female_tp, female_tn = 0, 0
    male_p, male_n = 0, 0
    female_p, female_n = 0, 0
    female = torch.tensor([1, 0]).type(torch.float32)
    male = torch.tensor([0, 1]).type(torch.float32)

    for i, (data, label) in enumerate(dataset):
        data = Variable(data)
        gt = Variable(label)
        label = torch.tensor([gt[0]])
        gender = gt[1:3]
        output = model(data)
        if torch.equal(gender, female) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                female_p += 1
                if pred == label.detach().numpy()[0]:
                    female_tp += 1
            else:
                pred = 0
                female_n += 1
                if pred == label.detach().numpy()[0]:
                    female_tn += 1
        elif torch.equal(gender, male) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                male_p += 1
                if pred == label.detach().numpy()[0]:
                    male_tp += 1
            else:
                pred = 0
                male_n += 1
                if pred == label.detach().numpy()[0]:
                    male_tn += 1
    # confusion matrix
    tp = {'female': female_tp, 'male': male_tp}
    tn = {'female': female_tn, 'male': male_tn}
    fp = {'female': female_p - female_tp, 'male': male_p - male_tp}
    fn = {'female': female_n - female_tn, 'male': male_n - male_tn}
    Attr = ['female', 'male']
    print("tp:", tp)
    print("tn:", tn)
    print("fp:", fp)
    print("fn:", fn)

    # normal accuracy
    female_acc = (female_tp + female_tn) / (female_p + female_n)
    male_acc = (male_tp + male_tn) / (male_p + male_n)
    total_acc = (female_tp + female_tn + male_tp + male_tn) / (female_p + female_n + male_p + male_n)
    acc = {'female': female_acc, 'male': male_acc}

    # equalized odds
    EO = Equalized_odds(tp, tn, fn, fp, Attr)

    return total_acc, EO


