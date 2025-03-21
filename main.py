import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F as f
from aiogram.filters.command import Command
import faiss
import pandas as pd

from utils import * # create_embedding, k_nn_faiss, list_texts, generate_promt, send_request_to_vllm, check_max_token_overflow

from base import add_info

from datetime import datetime

index = faiss.read_index("index.faiss")
df = pd.read_csv("catalog_page.csv", index_col = 0)
path = "md_docs"
api_url = "http://127.0.0.1:1240/v1/chat/completions"
api_url_token = "http://127.0.0.1:1240/tokenize" # серверное пространство для превращения текста в токены модели
model_name = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8"    # поменяла модель, так как другая перестала отвечать

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

# Объект бота
bot = Bot(token="your_bot_token")

# Диспетчер
dp = Dispatcher()

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Бот завода 'Энергокабель' приветствует Вас! Здесь вы можете найти полезную информацию по подбору кабеля. Задайте вопрос интересующий вас.")

# Хэндлер на команду /info
@dp.message(Command("info"))
async def cmd_start(message: types.Message):
    await message.answer("Это бот завода 'Энергокабель'! Бот поможет с подбором кабеля для Вас.")
    
# Хэндлер на команду /help
@dp.message(Command("help"))
async def cmd_start(message: types.Message):
    await message.answer("Бот является разработкой с внедрением искусственного интеллекта. Создан для подбора кабеля и ответов на вопросы про завод 'Энергокабель' и его продукцию. Задайте интересующий вопрос в чате.")

# Хэндлер на текстовые сообщения
@dp.message(f.text)
async def cmd_text(message: types.Message):
    query = message.text
    try:
        k = 15
        embedding = create_embedding(query)
        
        while k:
            I = k_nn_faiss(index, embedding, k = k)
            texts = list_texts(df, I, path)
            prompt = generate_promt(query, texts)
            if not check_max_token_overflow(prompt, api_url_token, model_name):
                break
            k = k-1
        print(k)
        answer = send_request_to_vllm(api_url, prompt, model_name)
        
        await message.answer(answer)
        
        add_info(user_name=message.from_user.first_name, timestamp=datetime.now(), query=query, all_error="это не ошибка", answer=answer)
    except Exception as e:
        logging.error(str(e))
        await message.answer(f"Произошла ошибка: {str(e)}")
        
        add_info(user_name=message.from_user.first_name, timestamp=datetime.now(), query=query, all_error="это ошибка", answer=str(e))
# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
