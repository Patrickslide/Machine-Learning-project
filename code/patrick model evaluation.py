import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\patri\\Desktop\\PATRICK\\Università\\Didattica\\Corsi\\Data Science\\Introduction to Machine Learning\\competition\\results_model4')

# get some query paths
qtest = list(df.columns)[:5]
TUO_PATH_PRIMA_DI_QUERY = 'C:\\Users\\patri\\Desktop\\PATRICK\\Università\\Didattica\\Corsi\\Data Science\\Introduction to Machine Learning\\DATA\\challenge_test_data\\query'
TUO_PATH_PRIMA_DI_GALLERY = 'C:\\Users\\patri\\Desktop\\PATRICK\\Università\\Didattica\\Corsi\\Data Science\\Introduction to Machine Learning\\DATA\\challenge_test_data\\gallery'

# plot some query images with top10
for queries in qtest:
    plt.figure(figsize=(20, 20))
    qpth = TUO_PATH_PRIMA_DI_QUERY
    qimg = plt.imread(qpth)
    ax = plt.subplot(3,4,1)
    plt.imshow(qimg)
    plt.title('query image')
    for i, j in zip(df[queries], range(10)):
        pth = TUO_PATH_PRIMA_DI_GALLERY
        img = plt.imread(pth)
        ax = plt.subplot(3,4,j+2)
        plt.imshow(img)
        plt.axis("off")
    plt.show()