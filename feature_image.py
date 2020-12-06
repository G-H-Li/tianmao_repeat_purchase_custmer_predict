import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    train = pd.read_csv(r".\dataset\train_data.csv")
    sns.countplot(x='age_range', order=[1, 2, 3, 4, 5, 6, 7, 8], hue='gender', data=train)
    plt.show()

    c = train.columns.tolist()

    train[c[5]].hist(range=[0, 80], bins=80)
    plt.xlabel(c[5])
    plt.ylabel("count")
    plt.show()

    train[c[6]].hist(range=[0, 40], bins=40)
    plt.xlabel(c[6])
    plt.ylabel("count")
    plt.show()

    train[c[7]].hist(range=[0, 10], bins=10)
    plt.xlabel(c[7])
    plt.ylabel("count")
    plt.show()

    train[c[8]].hist(range=[0, 10], bins=10)
    plt.xlabel(c[8])
    plt.ylabel("count")
    plt.show()

    train[c[9]].hist(range=[0, 50], bins=50)
    plt.xlabel(c[9])
    plt.ylabel("count")
    plt.show()

    train[c[10]].hist(range=[0, 3], bins=3)
    plt.xlabel(c[10])
    plt.ylabel("count")
    plt.show()

    train[c[11]].hist(range=[0, 6], bins=6)
    plt.xlabel(c[11])
    plt.ylabel("count")
    plt.show()

    train[c[12]].hist(range=[0, 6], bins=6)
    plt.xlabel(c[12])
    plt.ylabel("count")
    plt.show()

    train[c[13]].hist(range=[0, 6], bins=6)
    plt.xlabel(c[13])
    plt.ylabel("count")
    plt.show()
