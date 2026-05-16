import os                           # 환경변수 읽는 도구
from dotenv import load_dotenv      # .env 파일을 읽는 도구
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from base import Base

load_dotenv()                       # .env 파일에서 변수를 불러옴

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)


if __name__ == "__main__":
    import models
    try:
        engine.connect()
        print("DB 연결 성공")
        
        Base.metadata.create_all(engine)
        print("테이블 생성 완료")
        print(DATABASE_URL)
    except Exception as e:
        print(f"DB 연결 실패: {e}")