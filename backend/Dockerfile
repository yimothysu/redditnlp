FROM python:3.12
WORKDIR /code
COPY ./requirements_docker.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./ /code
CMD ["fastapi", "run", "main.py", "--port", "80"]