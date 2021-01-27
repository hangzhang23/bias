"""
result visualization skript, import excel files and visualize the corresponding results
"""
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def excel_processing(data):
    # clean all null blanks and combine the rest data together
    data1 = data.dropna(axis=1, how='all')
    data2 = data1.dropna(axis=0, how='any')

    # add two extra attributes in this dataframe for plotting works
    shape = ['Original', 'NoSex', 'No_Sex_Relationship', 'No_Sex_Relationship_Marital.Status']
    shape_list = []
    for i in range(6):
        shape_list = shape_list + shape


    data2 = data2.assign(classifier = ['LR','LR','LR','LR','LDA','LDA','LDA','LDA','KNN','KNN','KNN','KNN',
                                       'CART','CART','CART','CART','NB','NB','NB','NB','RF','RF','RF','RF'])
    data2 = data2.assign(shape=shape_list)

    return data2

def scatterplot(data):
    plt.figure()
    set1 = {"LR": "r", "LDA": "orange", "KNN": "lawngreen", "CART": "blueviolet", "NB": "fuchsia", "RF": "dodgerblue"}
    set2 = ["o", "s", "^", "X"]
    y_list = ["Statistical Parity", "Equalized Odds", "Predictive Parity", "Equal Opportunity",
              "Conditional use accuracy equality", "Overall accuracy equality", "Treatment equality"]

    for i in range(len(y_list)):
        g = sns.relplot(x="Accuracy", y=y_list[i], style="shape", hue="classifier", data=data, palette=set1,
                        markers=set2, legend=False)
        g.fig.suptitle("%s - Accuracy" % y_list[i])

    plt.show()


if __name__ == '__main__':
    data = pd.read_excel('Predictive Performance and Fairness.xlsx', nrows=29)
    plot_data = excel_processing(data)
    scatterplot(plot_data)

