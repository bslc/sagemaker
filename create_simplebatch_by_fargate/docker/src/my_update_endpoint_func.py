"""
対象の推論エンドポイントを更新する
"""

import pandas as pd
import json
import sagemaker
import boto3


def series_to_dict(ts, cat=None, dynamic_feat=None):
    """Given a pandas.Series object, returns a dictionary encoding the time series.

    ts -- a pands.Series object with the target time series
    cat -- an integer indicating the time series category

    Return value: a dictionary
    """
    obj = {"start": str(ts.index[0]), "target": encode_target(ts)}
    if cat is not None:
        obj["cat"] = cat
    if dynamic_feat is not None:
        obj["dynamic_feat"] = dynamic_feat
    return obj


class DeepARPredictor(sagemaker.predictor.RealTimePredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, content_type=sagemaker.content_types.CONTENT_TYPE_JSON, **kwargs)

    def predict(self, ts, cat=None, dynamic_feat=None,
                num_samples=100, return_samples=False, quantiles=["0.1", "0.5", "0.9"]):
        """Requests the prediction of for the time series listed in `ts`, each with the (optional)
        corresponding category listed in `cat`.

        ts -- `pandas.Series` object, the time series to predict
        cat -- integer, the group associated to the time series (default: None)
        num_samples -- integer, number of samples to compute at prediction time (default: 100)
        return_samples -- boolean indicating whether to include samples in the response (default: False)
        quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])

        Return value: list of `pandas.DataFrame` objects, each containing the predictions
        """
        prediction_time = ts.index[-1] + 1
        quantiles = [str(q) for q in quantiles]
        req = self.__encode_request(
            ts, cat, dynamic_feat, num_samples, return_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, ts.index.freq, prediction_time, return_samples)

    def __encode_request(self, ts, cat, dynamic_feat, num_samples, return_samples, quantiles):
        instance = series_to_dict(
            ts, cat if cat is not None else None, dynamic_feat if dynamic_feat else None)

        configuration = {
            "num_samples": num_samples,
            "output_types": ["quantiles", "samples"] if return_samples else ["quantiles"],
            "quantiles": quantiles
        }

        http_request_data = {
            "instances": [instance],
            "configuration": configuration
        }

        return json.dumps(http_request_data).encode('utf-8')

    def __decode_response(self, response, freq, prediction_time, return_samples):
        # we only sent one time series so we only receive one in return
        # however, if possible one will pass multiple time series as predictions will then be faster
        predictions = json.loads(response.decode('utf-8'))['predictions'][0]
        prediction_length = len(next(iter(predictions['quantiles'].values())))
        prediction_index = pd.DatetimeIndex(
            start=prediction_time, freq=freq, periods=prediction_length)
        if return_samples:
            dict_of_samples = {
                'sample_' + str(i): s for i, s in enumerate(predictions['samples'])}
        else:
            dict_of_samples = {}
        return pd.DataFrame(data={**predictions['quantiles'], **dict_of_samples}, index=prediction_index)


def _update_endpoint(estimator, DeepARPredictor, endpoint_name):

    # エンドポイントのリストを取得
    session = boto3.Session()
    sagemaker_client = session.client(
        'sagemaker', region_name='ap-northeast-1')
    response = sagemaker_client.list_endpoints()

    # 既に対象のエンドポイントが存在している場合はアップデート、存在しない場合は新規作成
    endpoint_name_list = []
    for endpoint in response['Endpoints']:
        endpoint_name_list.append(endpoint['EndpointName'])

    if endpoint_name in endpoint_name_list:
        print(10*'='+'start updating endpoint'+10*'=')
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge',
            predictor_cls=DeepARPredictor,
            endpoint_name=endpoint_name,
            update_endpoint=True,
            wait=True
        )
        print(10*'='+'finish updating endpoint'+10*'=')
    else:
        print(10*'='+'start creating endpoint'+10*'=')
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge',
            predictor_cls=DeepARPredictor,
            endpoint_name=endpoint_name,
            update_endpoint=False,
            wait=True
        )
        print(10*'='+'finish creating endpoint'+10*'=')

    return predictor


def main(estimator, endpoint_name):
    _update_endpoint(estimator, DeepARPredictor, endpoint_name)
