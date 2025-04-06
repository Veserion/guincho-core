from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

XGB_MODEL_PATH = "user_data/strategies/ml_regression/xgb_model.json"

# param_grid = {
#     'max_depth': [6, 8, 10, 12],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [300, 500],
#     'subsample': [0.5, 0.6, 0.8],
#     'colsample_bytree': [0.6, 0.8, 1.0]
#     # 'max_depth': [8],
#     # 'learning_rate': [0.05],
#     # 'n_estimators': [300],
#     # 'subsample': [0.6],
#     # 'colsample_bytree': [0.8]
# }
param_grid = {'colsample_bytree': [0.8], 'learning_rate': [0.05], 'max_depth': [9], 'n_estimators': [500], 'subsample': [0.8]}

def train_xgboost(X_train, y_train, X_val, y_val):
    grid = GridSearchCV(
        XGBRegressor(objective='reg:squarederror', eval_metric='rmse'),
        param_grid)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    print(f"best_params {best_params}")

    xgb_model = XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=10
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)
    xgb_model.save_model(XGB_MODEL_PATH)
    return xgb_model
