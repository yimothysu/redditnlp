FROM python:3.12
WORKDIR /code
COPY ./requirements/requirements_docker.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./data/ /code/data/
COPY ./src/ /code/src/
CMD ["fastapi", "run", "src/main.py", "--port", "80"]