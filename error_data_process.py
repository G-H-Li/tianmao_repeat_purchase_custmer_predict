import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


plt.rcParams["font.sans-serif"] = "SimHei"  # 解决中文乱码


if __name__ == "__main__":
    # read data
    train_data = pd.read_csv(r".\dataset\train_format1.csv")
    test_data = pd.read_csv(r".\dataset\test_format1.csv")
    user_info = pd.read_csv(r".\dataset\user_info_format1.csv")
    user_log = pd.read_csv(r".\dataset\user_log_format1.csv", dtype={'time_stamp': 'str'})
    print("data loaded!")
    print('-' * 50)

    # check and repair error data for user_info
    # user_info.info()
    print("start fix error data...")
    user_info['age_range'].replace(0.0, np.nan, inplace=True)
    user_info['gender'].replace(2.0, np.nan, inplace=True)
    # user_info.info()
    user_info['age_range'].replace(np.nan, -1, inplace=True)
    user_info['gender'].replace(np.nan, -1, inplace=True)
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
    print("start process train data to extract features...")
    # First: connect with age_range, gender
    train_data = pd.merge(train_data, user_info, on="user_id", how="left").fillna(method='ffill')
    train_data.head()
    train_data.info()
    # Second: connect with total_logs
    total_logs_tmp = user_log.groupby([user_log["user_id"], user_log["merchant_id"]])\
        .count()\
        .reset_index()[["user_id", "merchant_id", "item_id"]]
    total_logs_tmp.rename(columns={"item_id": "total_logs"}, inplace=True)
    train_data = pd.merge(train_data, total_logs_tmp, on=["user_id", "merchant_id"], how="left")
    train_data.head()
    train_data.info()
    # Third: connect with item_ids
    item_ids_tmp = user_log.groupby([user_log["user_id"], user_log["merchant_id"], user_log["item_id"]])\
        .count()\
        .reset_index()[["user_id", "merchant_id", "item_id"]]
    item_ids_tmp = item_ids_tmp.groupby([item_ids_tmp["user_id"], item_ids_tmp["merchant_id"]])\
        .count()\
        .reset_index()
    item_ids_tmp.rename(columns={"item_id": "item_ids"}, inplace=True)
    train_data = pd.merge(train_data, item_ids_tmp, on=["user_id", "merchant_id"], how="left")
    train_data.head()
    train_data.info()
    # Fourth: connect with category
    categories_temp = user_log.groupby([user_log["user_id"], user_log["merchant_id"], user_log["cat_id"]])\
        .count()\
        .reset_index()[["user_id", "merchant_id", "cat_id"]]
    categories_temp = categories_temp.groupby([categories_temp["user_id"], categories_temp["merchant_id"]])\
        .count()\
        .reset_index()
    categories_temp.rename(columns={"cat_id": "categories"}, inplace=True)
    train_data = pd.merge(train_data, categories_temp, on=["user_id", "merchant_id"], how="left")
    train_data.head()
    train_data.info()
    # Fifth: connect with browser_days
    browse_days_temp = user_log.groupby([user_log["user_id"], user_log["merchant_id"], user_log["time_stamp"]])\
        .count()\
        .reset_index()[["user_id", "merchant_id", "time_stamp"]]
    browse_days_temp = browse_days_temp.groupby([browse_days_temp["user_id"], browse_days_temp["merchant_id"]])\
        .count()\
        .reset_index()
    train_data = pd.merge(train_data, browse_days_temp, on=["user_id", "merchant_id"], how="left")
    train_data.head()
    train_data.info()
    # TODO: connect with the times per six months
    # Sixth: connect with click_count, shopping_carts, purchase_count, favourite_count
    one_clicks_temp = user_log.groupby([user_log["user_id"], user_log["merchant_id"], user_log["action_type"]])\
        .count()\
        .reset_index()[["user_id", "seller_id", "action_type", "item_id"]]
    one_clicks_temp.rename(columns={"item_id": "times"}, inplace=True)
    one_clicks_temp["click_count"] = (one_clicks_temp["action_type"] == 0) * one_clicks_temp["times"]
    one_clicks_temp["shopping_cart"] = (one_clicks_temp["action_type"] == 1) * one_clicks_temp["times"]
    one_clicks_temp["purchase_count"] = (one_clicks_temp["action_type"] == 2) * one_clicks_temp["times"]
    one_clicks_temp["favourite_count"] = (one_clicks_temp["action_type"] == 3) * one_clicks_temp["times"]
    four_features = one_clicks_temp.groupby([one_clicks_temp["user_id"], one_clicks_temp["merchant_id"]])\
        .sum()\
        .reset_index()
    four_features = four_features.drop(["action_type", "times"], axis=1)
    train_data = pd.merge(train_data, four_features, on=["user_id", "merchant_id"], how="left")
    train_data.head()
    train_data.info()
    print("feature extraction success!")
    print("saving train data...")
    train_data.to_csv(path_or_buf=r".\dataset\train_data.csv", index=False)
    print("saved")
    print('-' * 50)


