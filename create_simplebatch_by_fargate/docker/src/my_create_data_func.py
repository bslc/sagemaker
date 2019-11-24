"""
S3の参照元ファイルから、前処理を行う。　　
"""

import boto3
import pandas as pd
import numpy as np
import json


"""
set parametor
"""
# hyper parameters
prediction_length = 7 * 12
freq = '2H'
context_length = 7 * 12

# trainデータの期間を指定
start_dataset = pd.Timestamp("2014-01-01 00:00:00", freq=freq)
end_training = pd.Timestamp("2014-09-01 00:00:00", freq=freq)

# 利用するS3パス
s3_bucket = ""
s3_prefix = ""
s3_data_path = "s3://{}/{}/data".format(s3_bucket, s3_prefix)
s3_output_path = "s3://{}/{}/output".format(s3_bucket, s3_prefix)


# s3のプリフィックスを指定したら、そのKey以下のファイルを読み込む
def _read_s3_files(bucket: str, prefix: str):
    session = boto3.Session()
    s3_client = session.client('s3')

    df = pd.DataFrame()
    list_ = []
    response = s3_client.list_objects(
        Bucket=bucket,
        Delimiter=',',
        EncodingType='url',
        Prefix=prefix,
        RequestPayer='requester'
    )

    for key in response['Contents']:
        if key['Key'][-1] == '/':
            pass
        else:
            obj = s3_client.get_object(Bucket=bucket, Key=key['Key'])
            df = pd.read_csv(obj['Body'], sep=";", index_col=0,
                             parse_dates=True, decimal=',')

            list_.append(df)
    df = pd.concat(list_)

    return df


def _preprocess(df):
    num_timeseries = df.shape[1]
    data_kw = df.resample('2H').sum() / 8
    timeseries = []
    for i in range(num_timeseries):
        timeseries.append(np.trim_zeros(data_kw.iloc[:, i], trim='f'))

    return timeseries


def _create_train_test_data(df, start_dataset, end_training):
    num_test_windows = 4

    training_data = [
        {
            "start": str(start_dataset),
            "target": ts[start_dataset:end_training - 1].tolist()
        }
        for ts in df
    ]

    test_data = [
        {
            "start": str(start_dataset),
            "target": ts[start_dataset:end_training + k * prediction_length].tolist()
        }
        for k in range(1, num_test_windows + 1)
        for ts in df
    ]

    return training_data, test_data


def _write_dicts_to_file(path, data):
    with open(path, 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode('utf-8'))


def _copy_to_s3(local_file, s3_path, override=True):
    assert s3_path.startswith('s3://')

    s3 = boto3.resource('s3')

    split = s3_path.split('/')
    bucket = split[2]
    path = '/'.join(split[3:])
    buk = s3.Bucket(bucket)

    if len(list(buk.objects.filter(Prefix=path))) > 0:
        if not override:
            print(
                'File s3://{}/{} already exists.\nSet override to upload anyway.\n'.format(s3_bucket, s3_path))
            return
        else:
            print('Overwriting existing file')
    with open(local_file, 'rb') as data:
        print('Uploading file to {}'.format(s3_path))
        buk.put_object(Key=path, Body=data)


def main():
    print(10*'=' + 'start creating data' + 10*'=')

    df = _read_s3_files(bucket=s3_bucket, prefix=s3_prefix+'/raw')
    df = _preprocess(df=df)
    train_data, test_data = _create_train_test_data(
        df, start_dataset, end_training)

    # dataをローカルにファイルとして出力
    _write_dicts_to_file("train.json", train_data)
    _write_dicts_to_file("test.json", test_data)

    # fileをS3にコピー
    _copy_to_s3("train.json", s3_data_path + "/train/train.json")
    _copy_to_s3("test.json", s3_data_path + "/test/test.json")

    train_data_path = "{}/train/train.json".format(s3_data_path)
    test_data_path = "{}/test/test.json".format(s3_data_path)

    hyperparameters = {'prediction_length': prediction_length,
                       'freq': freq,
                       'context_length': context_length}
    print(10*'=' + 'creating data is finished' + 10*'=')

    return train_data_path, test_data_path, hyperparameters, s3_output_path
