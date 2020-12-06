import pandas as pd

if __name__ == "__main__":
    user_info = pd.read_csv(r".\dataset\user_info_format1.csv")
    user_info.info()
    user_log = pd.read_csv(r".\dataset\user_log_format1.csv")
    user_log.info()
    print(user_log["brand_id"].count())
