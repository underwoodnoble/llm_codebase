FROM nvcr.io/nvidia/pytorch:22.12-py3
WORKDIR /app
COPY dpo.txt /app/
RUN pip install -r dpo.txt
RUN apt-get update
RUN apt-get install -y openssh-server && apt-get install -y pssh && apt-get install -y tmux 

RUN service ssh start
EXPOSE 22