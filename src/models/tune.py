from xgboost import XGBClassifier
from pathlib import Path
import pandas as pd
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score
from sklearn.utils.class_weight import compute_sample_weight

models=Path('__file__').resolve().parent
SRC=models.parent
ROOT=SRC.parent

PATH_TRAIN_PARSED=ROOT/'data'/'parsed'/'train.csv'
PATH_TEST_PARSED=ROOT/'data'/'parsed'/'test.csv'
PATH_TRAIN_LABEL_PARSED=ROOT/'data'/'parsed'/'train_label.csv'
PATH_TEST_LABEL_PARSED=ROOT/'data'/'parsed'/'test_label.csv'

PATH_OPTUNA=ROOT/'data'/'model'/'optuna_data'
PATH_MODEL_PF=ROOT/'data'/'model'/'model_performance'
PATH_MODEL_PARS=ROOT/'data'/'model'/'model_optimal_parameters'

train_data=pd.read_csv(PATH_TRAIN_PARSED)
test_data=pd.read_csv(PATH_TEST_PARSED)
train_label=pd.read_csv(PATH_TRAIN_LABEL_PARSED).squeeze()
test_label=pd.read_csv(PATH_TEST_LABEL_PARSED).squeeze()


def run_optuna(train_data, train_label, n_trials):

    xgb_model=XGBClassifier(random_state=42)

    xgb_distributions={'n_estimators': IntDistribution(50, 300),
                        'learning_rate': FloatDistribution(0.01, 0.5, log=True),
                        'max_depth': IntDistribution(3, 10),
                        'subsample': FloatDistribution(0.5, 1.0),
                        'colsample_bytree': FloatDistribution(0.5, 1.0),
                        'gamma': FloatDistribution(0, 5),
                        'reg_alpha': FloatDistribution(1e-3, 1.0, log=True),
                        'reg_lambda': FloatDistribution(1.0, 11.0),}

    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_scorer=make_scorer(f1_score, average='macro')

    optuna_search=OptunaSearchCV(estimator=xgb_model,
                                param_distributions=xgb_distributions,
                                cv=cv,
                                scoring=recall_scorer,
                                n_trials=n_trials,
                                random_state=42)

    sample_weight=compute_sample_weight(class_weight='balanced', y=train_label)

    optuna_search.fit(train_data, train_label, sample_weight=sample_weight)

    return optuna_search.best_params_