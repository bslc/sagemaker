FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.4.1-cpu-py36-ubuntu16.04
COPY ./ /usr/src/app/
RUN pip install -r /usr/src/app/requirements.txt
WORKDIR /usr/src/app
ENV PYTHONPATH=/usr/src/app