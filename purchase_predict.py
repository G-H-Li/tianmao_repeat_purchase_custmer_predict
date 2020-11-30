import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# import seaborn as sns


plt.rcParams["font.sans-serif"] = "SimHei"  # 解决中文乱码


# data preprocessing
def preprocess():
    # read data
    print("start load data...")
    user_info = pd.read_csv(r".\dataset\user_info_format1.csv")
    user_log = pd.read_csv(r".\dataset\user_log_format1.csv", dtype={'time_stamp': 'str'})
    print("data loaded!")
    print('-' * 50)

    print("start process train data to extract features...")
    # check and repair error data for user_info
    # user_info.info()
    print("start fix error data...")
    user_info['age_range'].replace(0.0, np.nan, inplace=True)
    user_info['gender'].replace(2.0, np.nan, inplace=True)
    user_info.info()
    print("error data fix success!")
    print('-' * 50)
    # draw plots
    # user age-gender plot
    # sns.countplot(x='age_range', order=[-1, 1, 2, 3, 4, 5, 6, 7, 8], hue='gender', data=user_info)
    # user action plot
    # sns.countplot(x='action_type', order=[0, 1, 2, 3], data=user_log)

    # merge table to get feature
    # just to connect user_id, merchant_id with other param

    # Second: connect with total_logs
    total_logs_tmp = user_log.groupby([user_log["user_id"], user_log["seller_id"]]) \
        .count() \
        .reset_index()[["user_id", "seller_id", "item_id"]]
    total_logs_tmp.rename(columns={"seller_id": "merchant_id", "item_id": "total_logs"}, inplace=True)
    print("extract total_logs feature success!")
    print('-' * 50)

    # Third: connect with item_ids
    item_ids_tmp = user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["item_id"]]) \
        .count() \
        .reset_index()[["user_id", "seller_id", "item_id"]]
    item_ids_tmp = item_ids_tmp.groupby([item_ids_tmp["user_id"], item_ids_tmp["seller_id"]]) \
        .count() \
        .reset_index()
    item_ids_tmp.rename(columns={"seller_id": "merchant_id", "item_id": "item_ids"}, inplace=True)
    print("extract item_ids feature success!")
    print('-' * 50)

    # Fourth: connect with category
    categories_temp = user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["cat_id"]]) \
        .count() \
        .reset_index()[["user_id", "seller_id", "cat_id"]]
    categories_temp = categories_temp.groupby([categories_temp["user_id"], categories_temp["seller_id"]]) \
        .count() \
        .reset_index()
    categories_temp.rename(columns={"seller_id": "merchant_id", "cat_id": "categories"}, inplace=True)
    print("extract categories feature success!")
    print('-' * 50)

    # Fifth: connect with browser_days
    browse_days_temp = user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["time_stamp"]]) \
        .count() \
        .reset_index()[["user_id", "seller_id", "time_stamp"]]
    browse_days_temp = browse_days_temp.groupby([browse_days_temp["user_id"], browse_days_temp["seller_id"]]) \
        .count() \
        .reset_index()
    browse_days_temp.rename(columns={"seller_id": "merchant_id"}, inplace=True)
    print("extract browser_days feature success!")
    print('-' * 50)

    # Sixth: connect with click_count, shopping_carts, purchase_count, favourite_count
    one_clicks_temp = user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["action_type"]]) \
        .count() \
        .reset_index()[["user_id", "seller_id", "action_type", "item_id"]]
    one_clicks_temp.rename(columns={"seller_id": "merchant_id", "item_id": "times"}, inplace=True)
    one_clicks_temp["click_count"] = (one_clicks_temp["action_type"] == 0) * one_clicks_temp["times"]
    one_clicks_temp["shopping_cart"] = (one_clicks_temp["action_type"] == 1) * one_clicks_temp["times"]
    one_clicks_temp["purchase_count"] = (one_clicks_temp["action_type"] == 2) * one_clicks_temp["times"]
    one_clicks_temp["favourite_count"] = (one_clicks_temp["action_type"] == 3) * one_clicks_temp["times"]
    four_features = one_clicks_temp.groupby([one_clicks_temp["user_id"], one_clicks_temp["merchant_id"]]) \
        .sum() \
        .reset_index()
    four_features = four_features.drop(["action_type", "times"], axis=1)
    print("extract four_features feature success!")
    print('-' * 50)
    print("feature extraction success!")
    return user_info, total_logs_tmp, item_ids_tmp, categories_temp, browse_days_temp, four_features


def create_dataset(dataset_path, user_info, total_logs, item_ids, categories, browse_days, four_features, save_path):
    print("start create dataset...")
    data = pd.read_csv(dataset_path)
    data = pd.merge(data, user_info, on="user_id", how="left")
    data = pd.merge(data, total_logs, on=["user_id", "merchant_id"], how="left")
    data = pd.merge(data, item_ids, on=["user_id", "merchant_id"], how="left")
    data = pd.merge(data, categories, on=["user_id", "merchant_id"], how="left")
    data = pd.merge(data, browse_days, on=["user_id", "merchant_id"], how="left")
    data = pd.merge(data, four_features, on=["user_id", "merchant_id"], how="left")
    data = data.fillna(method="bfill").fillna(method="ffill")
    data.info()
    print("saving data...")
    data.to_csv(path_or_buf=save_path, index=False)
    print("saved")
    print('-' * 50)


def train_forest(data):
    X = data.drop(['user_id', 'merchant_id', 'label'], axis=1)
    Y = data['label']
    X.info()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=10)
    print("start train...")
    forest = RandomForestClassifier(n_estimators=10, random_state=2)
    forest.fit(x_train, y_train)
    predict_prob = forest.predict_proba(x_test)
    print(predict_prob[:])
    print("Accuracy on training set: {:.3f}".format(forest.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(forest.score(x_test, y_test)))
    return forest


def predict(data, model):
    x = data.drop(['user_id', 'merchant_id', 'prob'], axis=1)
    x.info()
    print("start predict...")
    predict_prob = model.predict_proba(x)
    data['prob'] = predict_prob[:, 1]
    choose = ['user_id', 'merchant_id', 'prob']
    result = data[choose]
    result.to_csv(path_or_buf=r".\dataset\prediction.csv", index=False)
    print("predict end!")


if __name__ == "__main__":
    train_path = r".\dataset\train_data.csv"
    test_path = r".\dataset\test_data.csv"
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        user_info, total_logs, item_ids, categories, browse_days, four_features = preprocess()
        if not os.path.exists(train_path):
            create_dataset(r".\dataset\train_format1.csv", user_info, total_logs, item_ids, categories, browse_days, four_features, train_path)
        if not os.path.exists(test_path):
            create_dataset(r".\dataset\test_format1.csv", user_info, total_logs, item_ids, categories, browse_days, four_features, test_path)

    print('load train data...')
    train_data = pd.read_csv(r".\dataset\train_data.csv")
    test_data = pd.read_csv(r".\dataset\test_data.csv")
    # start train
    model = train_forest(train_data)
    # start predict
    predict(test_data, model)

