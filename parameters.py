# =================== autoencoder ===================
RATIO_JSD = 0
# autoencoder training
AE_EPOCH = 400
AE_BATCH_SIZE = 4096
AE_LR = 0.002
FITTING_ITER = 250

PLOT_RESULT = './log/ae/train_201912051204_KLD_%s' % int(RATIO_JSD*10)
AE_MODEL_SAVE_PATH = './model/ae/KLD/KLD_%s' % RATIO_JSD

# encoded data
KDD_URL = "./data/kdd/census-income_preprocessed.csv"
FITTING_MODEL = './model/ae/KLD/KLD_%s/KLD_AE_%s_%s.pkl' % (RATIO_JSD, FITTING_ITER, AE_EPOCH)
ENCODED_DATA = "./data/kdd/encoded/kdd_encoded_KLD_%s.csv" % int(RATIO_JSD*10)

# =================== MLP classifier ================
RATIO_EO = 0
CLF_EPOCH = 3000
CLF_LR = 0.01
CLF_FITTING_ITER = 1000

SIG_THRESHOLD = 0.5

CLASSIFIER_SAVE_PATH = './model/classifier/KLD_%s_EO_%s' % (RATIO_JSD, RATIO_EO)
CLASSIFIER_RECORD_PATH = './log/classifier/KLD_%s_EO_%s_%s' % (RATIO_JSD, RATIO_EO, CLF_EPOCH)
CLASSIFIER_TESTSET_PATH = './data/kdd/encoded/kdd_encoded_KLD_0_%s_EO_%s_test.csv' % (int(RATIO_JSD*10), RATIO_EO)


# evaluate fitting model
CLF_FITTING_MODEL = './model/classifier/KLD_%s_EO_%s/EO_Classifier_%s_%s.pkl' % (RATIO_JSD, RATIO_EO, CLF_FITTING_ITER, CLF_EPOCH)

