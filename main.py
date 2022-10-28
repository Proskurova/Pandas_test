import pandas
from sklearn.ensemble import RandomForestClassifier

# Читаем эксель-файл
trips_data = pandas.read_excel("trips_data.xlsx")

# head - вывести строчки сверху
# trips_data.salary или trips_data["salary"] - обращение к колонке salary
# trips_data.salary.hist()  - вывод гистограммы зарплат это в калапсе
# describe - описать (набор данных, с т.з. статистики)
# value_counts - все уникальные значения и сколько раз они встречаются

# Выборка по условию
# (чуть похоже на SQL)
# trips_data[ условие ]
# Всех 50 летних любителей пляжного отдыха: trips_data[(trips_data.age == 50) & (trips_data.vacation_preference == "Пляжный отдых")]

# Machine Learning

# Цель: научиться по данным о клиенте предсказывать куда ему лучше полететь

# Обучаем модель на большем кол-ве примеров
# модель каким-то образом (математичкая магия) находить закономерности в данных
# X = Входные данные, то на основе чего мы делаем предсказания (salary,age, city,... )
# y / target = Выходные данные, то что мы предсказываем (target)
# Задача модели: рассмотреть множество примеров (где каждый пример это пара X, y)
# И научиться по X предсказывать y
# в Х всё кроме колонки target
# .drop выкинуть строчку axis=0 или колонку axis=1 из таблицы
X = trips_data.drop("target", axis=1)
y = trips_data.target
# .shape-сколько строчек и колонок

# get_dummies -превращает одну колонку с категорией в множество колонок
X_dummies = pandas.get_dummies(X, columns=["city", "vacation_preference", "transport_preference"])

# Выбрать модель?
# Библиотека? Sklearn?
# Варианты: Tensorflow, pytorch, keras, h2o, Spark ML, xgboost
# Пробовать разное: разные алгоритмы, разные настройки, разная обработка входных
# Data Science - это про эксперименты

# RandomForest- случайный лес деревьев решений
# Классификация - выбрать один из вариантов( Лондон, Париж ....)
rfc = RandomForestClassifier() # Можно указать настройка, даже желательно
# Обучаем модель
rfc.fit(X_dummies, y)
# Смотрим на сколько хорошо модель обучилась (обучила учебник = обучающая выборка).
print(rfc.score(X_dummies, y))
# Как получить предсказания? sklearn+pandas
# Пример данных
sample = {col: [0] for col in X_dummies.columns}
sample1 = {'salary': [130000],
 'age': [52],
 'family_members': [1],
 'city_Екатеринбург': [0],
 'city_Киев': [0],
 'city_Краснодар': [1],
 'city_Минск': [0],
 'city_Москва': [0],
 'city_Новосибирск': [0],
 'city_Омск': [0],
 'city_Петербург': [0],
 'city_Томск': [0],
 'city_Хабаровск': [0],
 'city_Ярославль': [0],
 'vacation_preference_Архитектура': [0],
 'vacation_preference_Ночные клубы': [0],
 'vacation_preference_Пляжный отдых': [0],
 'vacation_preference_Шоппинг': [1],
 'transport_preference_Автомобиль': [0],
 'transport_preference_Космический корабль': [0],
 'transport_preference_Морской транспорт': [0],
 'transport_preference_Поезд': [0],
 'transport_preference_Самолет': [1]}
example_df = pandas.DataFrame(data=sample1, columns=X_dummies.columns)
# Предсказание модели
print(rfc.predict(example_df))
# Что модель думает про вероятность других исходов
print(rfc.predict_proba(example_df)) # %
print(rfc.classes_) # Города

#1)Самых взрослых людей в каждом городе
# unique- уникальные элементы
# groupby- групировка данных
max_age = [trips_data.groupby(trips_data.city)["age"].max()]
# или print(trips_data.groupby(['city']).agg({'age': ['max']}))

#2)У кого из любителей Самолетов в среднем больше членов семьи
mean_Family_plane = [int((trips_data[trips_data.transport_preference == "Самолет"]).family_members.mean())]

#3)Кто предпочитает Архитектуру, люди с высокой зарплатой или с низкой
# новый столбик, который показывает из скольки частей состоит з/п, если одна часть=50000 round()-округление
trips_data["salary_band"] = (trips_data.salary / 50000).round()
# 1 - vacation_preference=архитектура
# 0 - vacation_preference=все остальное
trips_data["arch_lover"] = (trips_data.vacation_preference == 'Архитектура').astype(int)
trips_data.groupby('salary_band')['arch_lover'].sum()
#4) Какой диапазон возрастов (20-30, 30-40, 40-50, 50-60, 60-70, 70-80) имеет самую высокую среднюю зарплату
age_20_30 = (trips_data[(trips_data.age >= 20) & (trips_data.age <= 30)]).salary.max()
age_30_40 = (trips_data[(trips_data.age >= 30) & (trips_data.age <= 40)]).salary.max()
age_40_50 = (trips_data[(trips_data.age >= 40) & (trips_data.age <= 50)]).salary.max()
age_50_60 = (trips_data[(trips_data.age >= 50) & (trips_data.age <= 60)]).salary.max()
age_60_70 = (trips_data[(trips_data.age >= 60) & (trips_data.age <= 70)]).salary.max()
age_70_80 = (trips_data[(trips_data.age >= 70) & (trips_data.age <= 80)]).salary.max()
print(f"20-30={age_20_30}, 30-40={age_30_40}, 40-50={age_40_50}, 50-60={age_50_60}, 60-70={age_60_70}, 70-80={age_70_80}")
