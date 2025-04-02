import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(csv_path):
    # Загружаем CSV
    df = pd.read_csv(csv_path)

    # Проверяем наличие необходимых колонок
    if 'close' not in df.columns or 'predicted_price' not in df.columns:
        raise ValueError("CSV файл должен содержать колонки 'close' и 'predicted_price'")

    # Смещаем предсказание на одну строку вверх (влево по времени)
    df['shifted_predicted_price'] = df['predicted_price'].shift(2)

    # Убираем строки, где нет предсказания
    df = df.dropna(subset=['shifted_predicted_price'])

    # Вычисляем метрики
    mae = mean_absolute_error(df['close'], df['shifted_predicted_price'])
    mse = mean_squared_error(df['close'], df['shifted_predicted_price'])
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((df['close'] - df['shifted_predicted_price']) / df['close'])) * 100

    # Выводим результаты
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Построение графиков
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label="Реальная цена закрытия", color='blue')
    plt.plot(df.index, df['close'] - df['shifted_predicted_price']*df['close'], label="Предсказанная цена", color='red', linestyle='dashed')

    plt.xlabel("Индекс")
    plt.ylabel("Цена")
    plt.title("Сравнение реальной и предсказанной цены закрытия")
    plt.legend()
    plt.show()
# Пример использования
evaluate_predictions('user_data/csv/NEAR_USDT.csv')
