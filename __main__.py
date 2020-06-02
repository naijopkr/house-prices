import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from utils import wait_for_enter


def main():

    # Fetch data and print info
    df = pd.read_csv('data/USA_Housing.csv')

    print(df.head())
    wait_for_enter()

    print(df.info())
    wait_for_enter()

    print(df.describe())
    wait_for_enter()

    # EAD
    sns.pairplot(df)
    plt.savefig('output/pairplot.png')
    plt.clf()
    wait_for_enter()

    sns.distplot(df['Price'])
    plt.savefig('output/distplot_price.png')
    plt.clf()
    wait_for_enter()

    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.savefig('output/heatmap_corr.png')



if __name__ == '__main__':
    main()
