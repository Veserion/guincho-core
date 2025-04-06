from xgboost import XGBClassifier

def train_xgboost(X_train, y_train, X_val, y_val, sample_weights):
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.02,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)
    return model