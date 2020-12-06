import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf


if __name__ == "__main__":
    user_info = pd.read_csv(r".\dataset\user_info_format1.csv")
    user_info['age_range'].replace(0.0, np.nan, inplace=True)
    user_info['gender'].replace(2.0, np.nan, inplace=True)
    user_info['age_range'].replace(np.nan, -1, inplace=True)
    user_info['gender'].replace(np.nan, -1, inplace=True)
    sns.countplot(x='age_range', order=[-1, 1, 2, 3, 4, 5, 6, 7, 8], hue='gender', data=user_info)
    plt.show()
