import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


def predict_post_office_workload(new_data, model, scaler, label_encoder):
    """
    Функция предсказания загруженности почтового отделения.

    Args:
        new_data (pd.DataFrame): DataFrame с новыми данными для предсказания.
        model (LogisticRegression): Обученная модель логистической регрессии.
        scaler (StandardScaler): Объект для масштабирования данных.
        label_encoder (LabelEncoder): Объект для кодирования категориальных признаков.

    Returns:
        str: Предсказанная загруженность (низкая, средняя, высокая).
    """

    # Преобразовать категориальные признаки (День недели) в числовые
    new_data['Week_day'] = label_encoder.transform(new_data['Week_day'])

    # Масштабирование числовых признаков
    new_data = scaler.transform(new_data)

    predictions = model.predict(new_data)

    # Возврат предсказания
    return predictions[0]  # Возвращаем первое предсказание, так как мы предсказываем для одной строки

def predict_workload_by_day(data, model, scaler, label_encoder):
    days_of_week = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']

    workload_by_day = {}
    for day in days_of_week:
        # Получение средних значений для каждого дня недели
        day_data = data[data['Week_day'] == label_encoder.transform([day])[0]]
        average_visitors = day_data['Visitors_number'].mean()
        average_employees = day_data['Employees_number'].mean()
        new_data = pd.DataFrame({'Time': [1500],  #  Примерное время
                                'Week_day': [day],
                                'Visitors_number': [average_visitors],
                                'Employees_number': [average_employees]})

        # Предсказание загруженности
        prediction = predict_post_office_workload(new_data, model, scaler, label_encoder)

        workload_by_day[day] = prediction

    return workload_by_day

def predict_workload_by_time_and_day(data, model, scaler, label_encoder):

    # Список дней недели
    days_of_week = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']

    # Список временных интервалов (с шагом 1 часа)
    time_intervals = list(range(900, 2000, 100))

    workload_by_time_and_day = {}
    for day in days_of_week:
        # Цикл по временным интервалам
        for time_interval in time_intervals:
            # Получение средних значений для каждого временного интервала каждого дня недели
            filtered_data = data[
                (data['Week_day'] == label_encoder.transform([day])[0]) &
                (data['Time'] == time_interval)
            ]
            if not filtered_data.empty:
                average_visitors = filtered_data['Visitors_number'].mean()
                average_employees = filtered_data['Employees_number'].mean()

                # Создание фиктивных данных
                new_data = pd.DataFrame({'Time': [time_interval],
                                        'Week_day': [day],
                                        'Visitors_number': [average_visitors],
                                        'Employees_number': [average_employees]})

                # Предсказание загруженности
                prediction = predict_post_office_workload(new_data, model, scaler, label_encoder)

                workload_by_time_and_day[(day, f'{time_interval // 100}:{time_interval % 100:02}')] = prediction
            # else:
            #     print(f'Нет данных для {day} в {time_interval // 100}:{time_interval % 100:02}')

    return workload_by_time_and_day

data = pd.read_csv('post_office_data.csv',sep=';')
label_encoder = LabelEncoder()

data['Week_day'] = label_encoder.fit_transform(data['Week_day'])

X = data.drop('Workload', axis=1)
y = data['Workload']

scaler = StandardScaler()
X_train = scaler.fit_transform(X)

model = LogisticRegression(random_state=42)
model.fit(X_train, y)

# new_data = pd.DataFrame({'Time': [1500],
#                         'Week_day': ['Воскресенье'],
#                         'Visitors_number': [2],
#                         'Employees_number': [1]})
#
# predictions = predict_post_office_workload(new_data, model, scaler, label_encoder)
#
#
# print(f'Предсказания для ручного ввода параметров: {predictions}')
#
# workload_by_day = predict_workload_by_day(data, model, scaler, label_encoder)
# print(f'Предсказанная загруженность по дням недели: {workload_by_day}')
#
# workload_by_time_and_day = predict_workload_by_time_and_day(data, model, scaler, label_encoder)
# print(f'Предсказанная загруженность по дням недели и времени: {workload_by_time_and_day}')