import os
import requests
import numpy as np
import pandas as pd
import json
import faiss
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel

# Загрузка модели для embeddings
model = BGEM3FlagModel('BAAI/bge-m3',use_fp16 = True)


# Создание embeddings для запроса
def create_embedding(query):
    embeddings = model.encode(query,batch_size = 1, max_length = 512,)['dense_vecs']
    return embeddings


# Получаем ближайших соседей из индекса для запроса
def k_nn_faiss(index, query_embedding, k = 10):
    D, I = index.search(query_embedding.reshape((1, 1024)), k)
    return I.flatten()


# по индексам из faiss вытаскиваем сами страницы в строковом виде
def list_texts(df, list_index, path):
    selected_rows = df.iloc[list_index, :]
    list_text = []
    for index, row in selected_rows.iterrows():
        folder = row["folder"]
        folder_name = f"document_{folder}_html"
        page = row["file_name"]
        page_name = f"page_{page}.html"
        full_name = os.path.join(path, folder_name, "clean_html", page_name)
        with open (full_name, encoding="utf-8") as f:
            html = f.read()
            list_text.append(html)
    return list_text


# функция для создания промта из текста запроса и текста документа
def generate_promt(query, list_texts):
    prefix = "Запрос на получение информации."
    prefix += f"\nЗапрос: {query}\n"
    for text in list_texts:
        prefix += f"Документ:\n{text}\n"
    prefix += "На основе представленных документов и запроса пользователя сформулируй ответ, учитывающий ключевые моменты и содержимое документов.\nУчти, что найденные документы и части могут не отвечать на запрос. Если список документов пустой, то ответь что по запросу ничего не найдено. Если документы не релевантны запросу, то ответь на основании общих сведений. Представь ответ в виде обычного текста."
    return prefix


def send_request_to_vllm(api_url, prompt, model_name="default-model", max_tokens=200, temperature=0.4, headers=None, top_p = 0.7,
                         length_penalty = 1.2, repetition_penalty = 1.05, no_repeat_ngram_size = 3, do_sample = True, top_k = 100, stream = False):

    """
    Отправляет POST-запрос в VLLM сервис для генерации текста.

    :param api_url: URL API сервиса VLLM
    :param prompt: Текст запроса (промпт) для модели
    :param model_name: Имя модели (опционально)
    :param max_tokens: Максимальное количество токенов в ответе (опционально)
    :param temperature: Параметр температуры для генерации (опционально)
    :param headers: Заголовки запроса (опционально)
    :return: Ответ от сервиса в формате JSON
    """
    # Подготовка данных для отправки
    payload = {"model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
            "length_penalty": length_penalty,
            "early_stopping": True,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "do_sample": do_sample,
            "top_k": top_k,
            'stream': False  # Выключаем потоковую передачу
        }

    # Если заголовки не переданы, используем стандартные
    if headers is None:
        headers = {
            'Content-Type': 'application/json',
        }

    # Преобразуем данные в JSON
    json_data = json.dumps(payload)

    # Отправляем POST-запрос
    response = requests.post(api_url, data=json_data, headers=headers)

    # Проверяем статус ответа
    if response.status_code == 200:
        # Возвращаем ответ в формате JSON
        return response.json()["choices"][0]["message"]["content"]
    else:
        # В случае ошибки возвращаем статус код и текст ошибки
        raise ValueError ("Ошибка запроса")
    
    
def check_max_token_overflow(prompt, api_url_token, model_name):
    payload = {"model": model_name, "prompt": prompt}
    
    # Отправляем POST-запрос
    response = requests.post(api_url_token, json = payload)

    # Проверяем статус ответа
    if response.status_code == 200:
        # Возвращаем ответ в формате JSON
        json_response = response.json()
        count = json_response["count"]
        max_model_len =json_response["max_model_len"]
        return count > max_model_len
    else:
        # В случае ошибки возвращаем статус код и текст ошибки
        raise ValueError ("Ошибка запроса")