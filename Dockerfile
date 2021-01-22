FROM python:3.6.10

COPY requirements.txt /

RUN pip3 install -r /requirements.txt

COPY . /app

WORKDIR /app

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0", "main:app"]