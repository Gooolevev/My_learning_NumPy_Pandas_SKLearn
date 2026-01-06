# 🚀 Data Science Cheat Sheet: NumPy, Pandas, Scikit-Learn

## 🔢 NumPy (Работа с массивами)

```
import numpy as np
```

```
# Создание
np.array([1, 2, 3])          # Из списка
np.zeros((3, 4))             # Матрица нулей (3 строки, 4 столбца)
np.ones(5)                   # Вектор из единиц
np.arange(0, 10, 2)          # От 0 до 10 с шагом 2
np.linspace(0, 1, 5)         # 5 чисел от 0 до 1 с равным шагом
np.random.randint(0, 5, (10, 48) # Случайные числа [0, 4] + массив из 10 строк и 48 столбцов
np.where ('Condition', 'If yes','If no') # 'numpy if'

# Операции
arr.reshape(2, -1)           # Изменить форму (-1 вычисляется автоматически) (rows, col) 
arr.astype(np.float32)       # Изменить тип данных
np.dot(A, B) or A @ B        # Матричное умножение
arr.T                        # Транспонирование

# Агрегация
arr.mean(), arr.std()        # Среднее и стандартное отклонение
arr.sum(axis=0)              # Сумма по столбцам
np.argmin(arr)               # Индекс минимального элемента
```

## 🐼 Pandas (Таблицы и анализ)

```
import pandas as pd
```
```
df = pd.read_csv('data.csv')
```
```
f = pd.read_csv('data.csv')
df.head(10)                  # Первые 10 строк
df.info()                    # Типы данных и пропуски
df.describe()                # Статистика (mean, max, min...)
df.shape                     # (строки, колонки)

# Выбор данных
df['col_name']               # Выбрать колонку (Series)
df[['col1', 'col2']]         # Выбрать несколько колонок (DataFrame)
df.iloc[0:5, 0:3]            # Срез по индексам (строки 0-4, столбцы 0-2)
df.loc[df['age'] > 30]       # Фильтрация по условию

# Очистка и трансформация
df.dropna()                  # Удалить строки с NaN
df.fillna(value=0)           # Заполнить NaN нулями
df.drop('col', axis=1)       # Удалить колонку
df.rename(columns={'a':'b'}) # Переименовать
df['new'] = df['a'] * 10     # Создать колонку на лету

# Группировка
df.groupby('category')['price'].mean() # Средняя цена по категориям
df.pivot_table(index='a', columns='b', values='c') # Сводная таблица
```

## 🤖 Scikit-Learn (Машинное обучение)

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
```

```
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Только transform для теста!

# 4. Модель (Fit -> Predict)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# 5. Метрики
print(classification_report(y_test, predictions))
```
