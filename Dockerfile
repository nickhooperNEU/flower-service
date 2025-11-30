FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
COPY model_weights.pth .
EXPOSE 8080
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 main:app