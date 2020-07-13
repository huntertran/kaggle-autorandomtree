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

# target object
target_data = data["SalePrice"]

raw_features = []
features = []

meta_features = pd.read_csv(features_file_path)

for index, meta_feature in pd.DataFrame(meta_features).iterrows():
    if meta_feature[1] == 1:
        raw_features.append(meta_feature["feature_name"])

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

mea = mean_absolute_error(predictions, validation_value)

print("Validation MAE for Random Forest Model: {:,.0f}".format(mea))
