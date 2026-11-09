# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

# df=pd.read_csv("C:\\Users\\AADITHYA\\OneDrive\\Desktop\\Data\\house-prices-advanced-regression-techniques\\train.csv")

# none_cols = [
#     'Alley',
#     'BsmtQual',
#     'BsmtCond',
#     'BsmtExposure',
#     'BsmtFinType1',
#     'BsmtFinType2',
#     'FireplaceQu',
#     'GarageType',
#     'GarageFinish',
#     'GarageQual',
#     'GarageCond',
#     'PoolQC',
#     'Fence',
#     'MiscFeature'
# ]

# for col in none_cols:
#     df[col] = df[col].fillna('NA')

# df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
# df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())
# df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
# df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())
# # df['Qual_x_Area']   = df['OverallQual'] * df['GrLivArea']
# df['Area_per_Qual'] = df['GrLivArea'] / df['OverallQual']

# # df['TotalSF']       = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
# # df['TotalBath']     = (df['FullBath'] + 0.5 * df['HalfBath'] +
# #                        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
# # df['HouseAge']      = df['YrSold'] - df['YearBuilt']
# # df['RemodAge']      = df['YrSold'] - df['YearRemodAdd']
# # df['IsRemodeled']   = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)

# df.drop(columns=["Id"], inplace=True)


# quality_map = {
#     'NA': 0,
#     'Po': 1,
#     'Fa': 2,
#     'TA': 3,
#     'Gd': 4,
#     'Ex': 5
# }

# bsmt_exposure_map = {
#     'NA': 0,
#     'No': 1,
#     'Mn': 2,
#     'Av': 3,
#     'Gd': 4
# }

# bsmt_fin_map = {
#     'NA': 0,
#     'Unf': 1,
#     'LwQ': 2,
#     'Rec': 3,
#     'BLQ': 4,
#     'ALQ': 5,
#     'GLQ': 6
# }

# quality_cols = [
#     'ExterQual',
#     'ExterCond',
#     'BsmtQual',
#     'BsmtCond',
#     'HeatingQC',
#     'KitchenQual',
#     'FireplaceQu',
#     'GarageQual',
#     'GarageCond',
#     'PoolQC'
# ]

# for col in quality_cols:
#     df[col] = df[col].map(quality_map)

# df['BsmtExposure'] = df['BsmtExposure'].map(bsmt_exposure_map)

# df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmt_fin_map)
# df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmt_fin_map)


# df["CentralAir"] = df["CentralAir"].map({
#     "N": 0,
#     "Y": 1
# })



# cat_cols = df.select_dtypes(include='object').columns

# cat_cols = cat_cols.drop('SalePrice', errors='ignore')

# df = pd.get_dummies(
#     df,
#     columns=cat_cols,
#     drop_first=True,
#     dtype=int
# )


# X = df.drop("SalePrice", axis=1)
# y = df["SalePrice"]


# x_train, x_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )


# xgb = XGBRegressor(
#     n_estimators=2000,
#     learning_rate=0.01,
#     max_depth=3,
#     min_child_weight=3,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     gamma=0.1,
#     random_state=42
# )

# xgb.fit(x_train, y_train)

# pred = xgb.predict(x_test)


# mae = mean_absolute_error(y_test, pred)
# mse = mean_squared_error(y_test, pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, pred)

# print("===== XGBOOST RESULTS =====")
# print("MAE :", mae)
# print("MSE :", mse)
# print("RMSE:", rmse)
# print("R2  :", r2)




import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# 1. Load dataset
df=pd.read_csv("C:\\Users\\AADITHYA\\OneDrive\\Desktop\\Data\\house-prices-advanced-regression-techniques\\train.csv")

# 2. Select important features
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "YearBuilt",
    "Neighborhood"
]

X = df[features]
y = df["SalePrice"]

# 3. Separate numerical and categorical columns
num_cols = features[:-1]
cat_cols = ["Neighborhood"]

# 4. Preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 5. Create XGBoost model
xgb = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42
)

# 6. Complete ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb)
])

# 7. Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 8. Train model
model.fit(x_train, y_train)

# 9. Predictions
pred = model.predict(x_test)

# 10. Evaluation
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("===== XGBOOST RESULTS =====")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)

# 11. Save model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/xgb_model.pkl")

print("\nModel saved successfully!")



