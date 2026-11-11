FROM python:3.12-slim
WORKDIR /app
COPY requirements_light.txt .
RUN pip install --no-cache-dir -r requirements_light.txt
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
