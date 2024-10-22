import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from model import predict_post_office_workload, model, scaler, label_encoder, predict_workload_by_day, \
    predict_workload_by_time_and_day
from tts import generate_audio, ticketNum

app = FastAPI()


class Data(BaseModel):
    time: int
    weekday: list[str]
    visitor_numbers: list[int]
    employees_number: list[int]


# Функция предсказания для ручного ввода параметров
@app.post("/predict_post_office_workload")
async def predict_post_office_workload_f(data: Data):
    new_data = pd.DataFrame({'Time': [data.time],
                             'Week_day': data.weekday,
                             'Visitors_number': data.visitor_numbers,
                             'Employees_number': data.employees_number})
    predictions = predict_post_office_workload(new_data, model, scaler, label_encoder)
    return predictions

# Функция предсказания по дням недели
@app.post("/predict_workload_by_day")
async def get_workload_by_day_f(data: Data):
    new_data = pd.DataFrame({'Time': [data.time],
                             'Week_day': data.weekday,
                             'Visitors_number': data.visitor_numbers,
                             'Employees_number': data.employees_number})
    predictions = predict_workload_by_day(data, model, scaler, label_encoder)
    return predictions


@app.post("/predict_workload_by_time_and_day")
async def predict_workload_by_time_and_day_f(data: Data):
    new_data = pd.DataFrame({'Time': [data.time],
                             'Week_day': data.weekday,
                             'Visitors_number': data.visitor_numbers,
                             'Employees_number': data.employees_number})
    predictions = predict_workload_by_time_and_day(data, model, scaler, label_encoder)
    return predictions


@app.post("/audio", summary="Получить аудио")
async def audio(ticket_num: str, window_num: int):
    path = generate_audio(ticket_num, window_num)
    return path


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(8006)
    )