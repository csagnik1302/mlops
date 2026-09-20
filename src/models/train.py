from xgboost import XGBClassifier
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import os
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
ROOT = Path(__file__).resolve().parent.parent.parent
mlflow.set_tracking_uri(f"file:///{ROOT}/mlruns")


def train(x_train, y_train, x_test, y_test, train_dataparms):

    sample_weight=compute_sample_weight(class_weight='balanced', y=y_train)

    model=XGBClassifier(**train_dataparms)

    with mlflow.start_run():
        model.fit(x_train, y_train, sample_weight=sample_weight) 
        y_pred=model.predict(x_test)
        
        accuracy=accuracy_score(y_test, y_pred)
        recall=recall_score(y_test, y_pred, average='macro')
        f1=f1_score(y_test, y_pred, average='macro')

        signature = infer_signature(x_train, y_pred)
        mlflow.log_params(train_dataparms)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        mlflow.xgboost.log_model(model, 'model', signature=signature)

        # log dataset for ui
        train_ds=mlflow.data.from_pandas(x_train, source='train_data')
        mlflow.log_input(train_ds, context='training')

        print('training complete')

    return model


