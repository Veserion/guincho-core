import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import os

from data_loader import load_data, split_data
from models.xgboost_model import train_xgboost, XGB_MODEL_PATH
from config import TIME_STEPS
from utils import create_sequences, flatten_sequences

# 1. Загрузка и подготовка данных
df, features = load_data()
train_df, val_df, test_df = split_data(df)

# 2. Создание последовательностей
X_train, y_train = create_sequences(train_df[features], train_df['target'], TIME_STEPS)
X_val, y_val = create_sequences(val_df[features], val_df['target'], TIME_STEPS)
X_test, y_test = create_sequences(test_df[features], test_df['target'], TIME_STEPS)

# XGBoost требует flat-данные
X_train = flatten_sequences(X_train)
X_val = flatten_sequences(X_val)
X_test = flatten_sequences(X_test)

# 3. Загрузка или обучение модели
if os.path.exists(XGB_MODEL_PATH):
    xgb_model = XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)
    print("Модель XGBoost загружена из файла.")
else:
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    print("Модель XGBoost обучена и сохранена в файл.")

# 4. Предсказание
y_pred = xgb_model.predict(X_test)

# 5. Метрики качества
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n===== Regression Metrics on Test Set =====")
print(f"MAE  (Mean Absolute Error):      {mae:.6f}")
print(f"MSE  (Mean Squared Error):       {mse:.6f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.6f}")
print(f"R²   (R-squared):                {r2:.6f}")

# 6. Построение графика
close_test = test_df['truth_close'].values[TIME_STEPS:]

plt.figure(figsize=(14, 6))
plt.plot(close_test[:100], label='Close', color='gray', linewidth=1)
plt.plot(y_pred[:100], label='Predicted', linewidth=1)
plt.title('XGBoost Prediction vs Close on Test Set')
plt.xlabel('Samples')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()