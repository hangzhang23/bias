import torch
import parameters as par
from Dataset_maker import Dataset_maker_classification
from functions.classifier_metrics import balanced_accuracy_gender, accuracy_calculate_gender


# hyper parameter
print("JSD:%s, EO:%s" % (par.RATIO_JSD, par.RATIO_EO))
URL = par.CLASSIFIER_TESTSET_PATH
ClASSIFIER_PATH = par.CLF_FITTING_MODEL

# read dataset
dataset_test = Dataset_maker_classification(URL)

# load classifier
model = torch.load(ClASSIFIER_PATH)
balanced_accuracy_gender(dataset_test, model)