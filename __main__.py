import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils import wait_for_enter

def print_and_wait(*output):
    print(*output)
    wait_for_enter()

def main():

    # Fetch data and print info
    df = pd.read_csv('data/USA_Housing.csv')

    print_and_wait(df.head())

    print_and_wait(df.info())

    print_and_wait(df.describe())


    # EAD
    sns.pairplot(df)
    plt.savefig('output/pairplot.png')
    plt.clf()

    sns.distplot(df['Price'])
    plt.savefig('output/distplot_price.png')
    plt.clf()

    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.savefig('output/heatmap_corr.png')
    plt.clf()


    # Training a Linear Regression Model
    X = df[df.columns[:-2]]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

    lm = LinearRegression()
    lm.fit(X_train, y_train)


    # Model evaluation
    print_and_wait(lm.intercept_)

    print_and_wait(lm.coef_)

    coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
    print_and_wait(coeff_df)


    # Predictions
    predictions = lm.predict(X_test)

    plt.scatter(y_test, predictions)
    plt.savefig('output/ytest_vs_predictions.png')
    plt.clf()

    sns.distplot(y_test - predictions, bins=50)
    plt.savefig('output/error_diff.png')
    plt.clf()


    # Metrics
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print_and_wait('MAE: ', mae)
    print_and_wait('MSE: ', mse)
    print_and_wait('RMSE: ', rmse)


if __name__ == '__main__':
    main()
