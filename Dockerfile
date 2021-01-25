FROM python:3.6.10

COPY requirements.txt /

RUN pip3 install -r /requirements.txt

COPY . /app

EXPOSE 8000

WORKDIR /app

ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "sentiment_analyzer.api:app", "--reload"]
