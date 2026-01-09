import pandas as pd
import numpy as np

np.random.seed(42)

n = 100000

df = pd.DataFrame({
    'ID': range(1, n + 1),
    'Отдел': np.random.choice(['IT', 'HR', 'Sales'], n),
    'Зарплата': np.random.randint(50000, 150000, n),
    'Опыт': np.random.uniform(1, 10, n).round(1)
})


df.to_csv('data.csv', index=False)

new_df = pd.read_csv('data.csv')
print(new_df.head())