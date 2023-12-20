from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose  import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def QuickEvaluate(train_df,test_df,target_col,num_col,cat_col):
  # Scale/Impute numerical columns, 1-hot encode categorical variables
  num_pipeline_list = [("imputer",SimpleImputer(strategy = "median")),
                        ("scaler",StandardScaler())]
  num_pipeline = Pipeline(num_pipeline_list)

  full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_col),
    ("cat",OneHotEncoder(sparse=False),cat_col)  
    ])
  
  full_pipeline.fit(train_df)
  train_data =full_pipeline.transform(train_df)
  test_data = full_pipeline.transform(test_df)

  model = LinearRegression()
  model.fit(train_data,train_df[target_col])

  result = model.predict(test_data)
  mse = mean_squared_error(test_df[target_col], result,squared=False)
  return mse
