FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN mkdir -p /fget
WORKDIR /fget

COPY . /fget

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install -U pip

RUN pip install jupyter-client>=7.0.0
RUN pip install jupyter-console
RUN pip install hydra-core --upgrade
RUN pip install fastapi pandas requests fastparquet && pip install "uvicorn[standard]"
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]