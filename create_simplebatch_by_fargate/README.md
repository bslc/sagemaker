## 概要
「ECS Scheduled Tasks (cron)」でFargateを定時実行させるテンプレートファイルです。  
「bash/main.sh」と「docker/src/my_create_data_func.py」と「docker/main.py」に変数を指定した上で「main.sh」を実行することで利用できます。


## やっていること
1.「ECS Scheduled Tasks (cron)」でJOBを起動  
2.Fargateで「s3からファイルを取得して前処理」→「SageMakerでのモデル学習」→「推論エンドポイント更新」


## 指定するべきもの
- bash/main.sh
    - STACK_NAME
        - テンプレートのスタック名です
        - その他諸々のリソース名もこれに合わせています
        - ex.yoshim-simple-batch-fargate
    - S3_BUCKET_NAME
        - 利用するS3バケット名です
        - ex.sagemaker-my-bucket
    - AWS_REGION
        - 対象のリージョンです
        - ex.ap-northeast-1
    - AWS_PROFILE
        - awscliの名前付きプロファイル名です
        - ローカルからデプロイすることを想定しています
        - ex.cm-yoshim-profile
- docker/src/my_create_data_func.py
    - s3_bucket
        - 利用するS3バケット名です
        - ex.sagemaker-my-bucket
    - s3_prefix
        - 利用するS3パスのプレフィックスです
        - yoshim-simple-batch-fargate/my-job
- docker/main.py
    - endpoint_name
        - SageMaker推論エンドポイント名です


## その他の留意点
- IAMロールの権限
    - 適宜必要なものに変更してください
- ネットワーク構成
    - 適宜必要なものに変更してください
    - 今回はパブリックサブネットにしています
- 今回の処理内容
    - SageMakerの[サンプル](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/deepar_electricity/DeepAR-Electricity.ipynb)をそのまま実行しています
    - 実際に使う場合は、当然ですが処理内容を書き換える必要があります
- デプロイ方法
    - 「create_simplebatch_by_fargate」直下で「bash ./bash/main.sh」を実行
