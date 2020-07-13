import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# read from the features list
features_file_path = "./configs/features.csv"
data_file_path = "./data/train.csv"

# data object
data = pd.read_csv(data_file_path)
dataFrame = pd.DataFrame(data)

meta_features = pd.read_csv(features_file_path)
meta_features_dataframe = pd.DataFrame(meta_features)

def find_mae():
    # target object
    target_data = data["SalePrice"]

    raw_features = []
    features = []

    for meta_feature in meta_features_dataframe.iterrows():
        if meta_feature[1]['is_include'] == 1:
            raw_features.append(meta_feature[1]["feature_name"])

    # transform category data
    encoder = LabelEncoder()

    for feature in raw_features:
        # print("Column: " + feature + " | " + "Data type: " + str(dataFrame[feature].dtypes))
        if dataFrame[feature].dtypes is not int:
            # category data
            new_feature_name = feature + "_encoded"
            data[new_feature_name] = encoder.fit_transform(data[feature].astype(str))
            features.append(new_feature_name)

    train_data = data[features]

    # split train_data to sub training data and validation data
    sub_training_data, sub_value, validation_data, validation_value = train_test_split(
        train_data, target_data, random_state=1
    )

    # specify model
    model = RandomForestRegressor(random_state=1)
    model.fit(sub_training_data, validation_data)
    predictions = model.predict(sub_value)

    mae = mean_absolute_error(predictions, validation_value)

    print("Validation MAE for Random Forest Model: {:,.0f}".format(mae))

    return mae

def find_features():
    mae = -1

    row_count = len(meta_features_dataframe)
    index = 0

    while index < row_count:
        if meta_features_dataframe.loc[index, 'is_checked'] == 0:
            # do the work
            meta_features_dataframe.at[index, 'is_include'] = 1
            new_mae = find_mae()
            if mae == -1:
                mae = new_mae
            elif new_mae <= mae:
                mae = new_mae
            else:
                meta_features_dataframe.at[index, 'is_include'] = 0
            
            meta_features_dataframe.loc[index, 'is_checked'] = 1
            meta_features_dataframe.to_csv(features_file_path, index=False)

        print("LOCKED MAE: {:,.0f}".format(mae))
        index = index + 1

find_features()