from src import my_create_data_func, my_train_func, my_update_endpoint_func

# your service's endpoint name
endpoint_name = ''

train_data_path, test_data_path, hyperparameters, s3_output_path = my_create_data_func.main()
estimator = my_train_func.main(
    train_data_path, test_data_path, hyperparameters, s3_output_path)
my_update_endpoint_func.main(estimator, endpoint_name=endpoint_name)
