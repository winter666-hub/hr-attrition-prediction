# Docker Hub에서 Python 3.11 slim 이미지를 베이스로 사용
# slim = 불필요한 패키지 제거된 가벼운 버전
FROM python:3.11-slim

WORKDIR /app

# 로컬의 requirements.txt를 컨테이너로 복사
COPY requirements.txt .

# 패키지 설치 (빌드 시 1번만 실행)
RUN pip install -r requirements.txt

COPY . .

# 컨테이너 실행 시 uvicorn으로 FastAPI 서버 시작
# 0.0.0.0 = 외부에서 접속 가능하게
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]