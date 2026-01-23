FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (OpenCV 등) - 생략 (opencv-python-headless 사용 시 불필요할 수 있음)
# RUN ... (네트워크 에러로 인해 스킵 시도)

# 요구사항 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# Flask 환경 변수 설정 (기본값)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 포트 노출
EXPOSE 5000

# 실행 명령
CMD ["flask", "run"]
