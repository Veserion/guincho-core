import pandas as pd
import matplotlib.pyplot as plt

# Загружаем df из csv
df = pd.read_csv('user_data/csv/NEAR_USDT_predicted.csv', parse_dates=True, index_col=0)

plt.figure(figsize=(15, 6))

plt.plot(df.index, df['close'], label='Close', color='blue')
# plt.plot(df.index, df['target'], label='Target', color='orange')
plt.plot(df.index, df['predicted_sma10'], label='predicted_sma10', color='red')

plt.legend()
plt.grid(True)
plt.title('Close vs Target')
plt.xlabel('Time')
plt.ylabel('Price')

plt.tight_layout()
plt.show()