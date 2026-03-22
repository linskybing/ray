FROM rayproject/ray:2.46.0-py312-gpu

WORKDIR /home/ray/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH="/home/ray/app:${PYTHONPATH}"
