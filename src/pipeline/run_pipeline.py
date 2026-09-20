from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.models.train import train
from src.models.evaluate import evaluate
import tomllib

with open('config.toml','rb') as f:
    config=tomllib.load(f)

test_size=config['preprocess']['test_size']
multicoll_corr_threshold=config['preprocess']['multicoll_corr_threshold']
random_state=config['random_state']['random_state']


def main(raw_data_path, train_params, test_size_m=test_size, multicoll_corr_threshold_m=multicoll_corr_threshold, random_state_m=random_state):

    data=load_data(raw_data_path)
    data_train, data_test, label_train, label_test=preprocess(data_train=data, test_size=test_size_m, random_state=random_state_m, multicoll_corr_threshold=multicoll_corr_threshold_m)

    # train
    model=train(x_train=data_train, y_train=label_train, x_test=data_test, y_test=label_test, train_dataparms=train_params)
    evaluate(model=model, x_test=data_test, y_test=label_test)



if __name__=='__main__':

    from pathlib import Path
    import json

    pipeline=Path(__file__).resolve().parent
    SRC=pipeline.parent
    ROOT=SRC.parent

    DATA_PATH=ROOT/'data'/'raw'/'train.csv'
    PARAMS_PATH=ROOT/'data'/'model'/'model_optimal_parameters'/'xgboost.json'

    with open(PARAMS_PATH, 'r') as f:
        params=json.load(f)

    main(raw_data_path=DATA_PATH, train_params=params)




