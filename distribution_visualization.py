import pandas as pd
import matplotlib.pyplot as plt

DATA_URL = './data/kdd/encoded/kdd_encoded_JSD_0.csv'
data1 = pd.read_csv(DATA_URL)

# df1 and df2
plt.figure(figsize=(12.8, 7.2))
for m in range(10):
    plt.subplot(4, 3, m+1)

    f_female = data1['feature_%s' % (m + 1)][data1.sex == 'Female']
    f_male = data1['feature_%s' % (m + 1)][data1.sex == 'Male']
    plt.hist(f_female, bins=20, label='female', color='steelblue', alpha=0.7)
    plt.hist(f_male, bins=20, label='male', color='red', alpha=0.6)
    plt.title('distribution in feature_%s' % (m + 1))
    plt.legend()

# plt.savefig('./log/JSD_1.png')
plt.show()
