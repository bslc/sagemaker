#!/bin/bash

set -e -o pipefail

# 環境変数の指定
export STACK_NAME=""
export S3_BUCKET_NAME=""
export AWS_REGION=""
export AWS_PROFILE=""
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --profile ${AWS_PROFILE} | jq -r '.Account')

# Fargateに必要なリソースをcfnテンプレートとしてデプロイ
echo "========== start deploy template =========="
aws cloudformation deploy \
--stack-name ${STACK_NAME}-stack \
--template-file templates/stack-for-fargate.yml \
--no-fail-on-empty-changeset \
--parameter-overrides StackName=${STACK_NAME} AwsAccountId=${AWS_ACCOUNT_ID} Region=${AWS_REGION} \
--capabilities CAPABILITY_NAMED_IAM \
--region ${AWS_REGION} --profile ${AWS_PROFILE}
echo "========== finish deploy template =========="

# ecrリポジトリにイメージをPUSH
echo "========== start pull mxnet image =========="
bash ./bash/pull.sh
echo "========== finish pull mxnet image =========="

echo "========== start build docker image =========="
bash ./bash/build.sh
echo "========== finish docker image build =========="

echo "========== start push docker image =========="
bash ./bash/push.sh
echo "========== finish push docker image =========="