"""
「my_create_data_func.py」で作成したファイルを元にSageMakerの学習ジョブを実行する。
"""

import sagemaker

sagemaker_session = sagemaker.Session()
image_name = sagemaker.amazon.amazon_estimator.get_image_uri(
    "ap-northeast-1", "forecasting-deepar", "latest")
role = sagemaker.get_execution_role()


def _get_estimator(s3_output_path: str):
    estimator = sagemaker.estimator.Estimator(
        sagemaker_session=sagemaker_session,
        image_name=image_name,
        role=role,
        train_instance_count=1,
        train_instance_type='ml.c4.2xlarge',
        base_job_name='deepar-electricity-demo',
        output_path=s3_output_path
    )

    return estimator


def _set_hyperparameters(estimator: sagemaker.estimator.Estimator, hp):
    freq = hp['freq']
    context_length = hp['context_length']
    prediction_length = hp['prediction_length']

    hyperparameters = {
        "time_freq": freq,
        "epochs": "40",
        "early_stopping_patience": "40",
        "mini_batch_size": "64",
        "learning_rate": "5E-4",
        "context_length": str(context_length),
        "prediction_length": str(prediction_length)
    }

    estimator.set_hyperparameters(**hyperparameters)

    return estimator


def _train(estimator: sagemaker.estimator.Estimator, train_data_path: str, test_data_path: str):
    data_channels = {
        "train": "{}".format(train_data_path),
        "test": "{}".format(test_data_path)
    }

    estimator.fit(inputs=data_channels, wait=True)
    return estimator


def main(train_data_path: str, test_data_path: str, hyperparameters, s3_output_path: str):

    estimator = _get_estimator(s3_output_path)
    estimator = _set_hyperparameters(estimator, hyperparameters)

    print(10*'=' + 'start training-job' + 10*'=')
    estimator = _train(estimator, train_data_path, test_data_path)
    print(10*'=' + 'training-job is finished' + 10*'=')

    return estimator
