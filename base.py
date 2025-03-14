from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Движок работы с базой данных
engine = create_engine("sqlite:///base_logs.db")

# Класс для создания таблиц
class Base(DeclarativeBase): pass

# Класс таблицы
class Log(Base):
    __tablename__ = "cabel_logs"
    id = Column(Integer, primary_key = True, index=True, autoincrement=True)
    user_name = Column(String, nullable=False, default="Иван Петров")
    timestamp = Column(DateTime)
    query = Column(Text, nullable=False)
    all_error = Column(String, nullable=False)
    answer = Column(Text, nullable=False)
    
# Создаем таблицу
Base.metadata.create_all(bind=engine)

# Создаем класс сессий
Session = sessionmaker(autoflush=False, bind=engine)

# Добавление информации к базе
def add_info(user_name, timestamp, query, all_error, answer):
    with Session(autoflush=False, bind=engine) as db:
        log = Log(user_name=user_name, timestamp=timestamp, query=query, all_error=all_error, answer=answer)
        db.add(log)
        db.commit() # Сохраняем изменения в БД