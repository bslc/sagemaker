export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --profile ${AWS_PROFILE} | jq -r '.Account')

$(aws ecr get-login --no-include-email --region ${AWS_REGION} --profile ${AWS_PROFILE} --registry-ids ${AWS_ACCOUNT_ID})
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${STACK_NAME}:latest