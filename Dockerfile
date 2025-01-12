FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /docker

COPY prototyping /docker

RUN pip install keyboard pynput opencv-python

CMD ["python" "/prototyping"]
