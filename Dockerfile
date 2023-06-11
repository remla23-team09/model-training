FROM python:3.9-slim as packages

WORKDIR /model-training
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy Data
FROM packages as data

COPY ./data/ ./data/

COPY .dvc/ /./dvc/
COPY ./models/ /./models/
COPY dvc.lock .

# Run the app
FROM data as app

COPY .git/ ./.git/
COPY src ./src

EXPOSE 8080

WORKDIR /model-training/src/web_interface
CMD ["python", "web_interface.py"]