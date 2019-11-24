# AWSが提供しているイメージをPULL
$(aws ecr get-login --no-include-email --region us-east-1 --registry-ids 763104351884 --profile ${AWS_PROFILE}) 
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.4.1-cpu-py36-ubuntu16.04