import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess(data_train, test_size, random_state, multicoll_corr_threshold):

    # #########################################################################
    # 1. Train / Test Split
    # #########################################################################
    data_train, data_test = train_test_split(data_train,
                                            test_size=test_size,
                                            random_state=random_state,
                                            stratify=data_train['churn_risk_score'])

    # #########################################################################
    # 2. Drop Unnecessary Columns & Rename Columns
    # #########################################################################
    data_train = data_train.drop(labels=['customer_id', 'Name', 'security_no', 'referral_id', 'last_visit_time'], axis=1)
    data_train = data_train.rename(columns={'avg_time_spent': 'avg_time_spent_in_seconds'})

    data_test = data_test.drop(labels=['customer_id', 'Name', 'security_no', 'referral_id', 'last_visit_time'], axis=1)
    data_test = data_test.rename(columns={'avg_time_spent': 'avg_time_spent_in_seconds'})

    # #########################################################################
    # 3. Handle Placeholder & Invalid Values
    # #########################################################################
    data_train['days_since_last_login'] = data_train['days_since_last_login'].replace(-999, pd.NA)
    data_train.loc[data_train['avg_time_spent_in_seconds'] < 0, 'avg_time_spent_in_seconds'] = pd.NA
    data_train.loc[data_train['points_in_wallet'] < 0, 'points_in_wallet'] = pd.NA
    data_train.loc[data_train['avg_frequency_login_days'] == 'Error', 'avg_frequency_login_days'] = pd.NA
    data_train.loc[data_train['churn_risk_score'] < 0, 'churn_risk_score'] = pd.NA

    data_test['days_since_last_login'] = data_test['days_since_last_login'].replace(-999, pd.NA)
    data_test.loc[data_test['avg_time_spent_in_seconds'] < 0,
                            'avg_time_spent_in_seconds'] = pd.NA
    data_test.loc[data_test['points_in_wallet'] < 0, 'points_in_wallet'] = pd.NA
    data_test.loc[data_test['avg_frequency_login_days'] == 'Error',
                            'avg_frequency_login_days'] = pd.NA
    data_test.loc[data_test['churn_risk_score'] < 0, 'churn_risk_score'] = pd.NA

    # #########################################################################
    # 4. Data Type Conversions
    # #########################################################################
    data_train['joining_date'] = pd.to_datetime(data_train['joining_date'], errors='coerce')
    data_train['avg_frequency_login_days'] = pd.to_numeric(data_train['avg_frequency_login_days'], errors='coerce')
    data_train['days_since_last_login'] = pd.to_numeric(data_train['days_since_last_login'], errors='coerce')

    data_train.loc[data_train['avg_frequency_login_days'] < 0,
                                'avg_frequency_login_days'] = pd.NA

    data_test['joining_date'] = pd.to_datetime(data_test['joining_date'], errors='coerce')
    data_test['avg_frequency_login_days'] = pd.to_numeric(data_test['avg_frequency_login_days'], errors='coerce')
    data_test['days_since_last_login'] = pd.to_numeric(data_test['days_since_last_login'], errors='coerce')

    data_test.loc[data_test['avg_frequency_login_days'] < 0,
                'avg_frequency_login_days'] = pd.NA

    # #########################################################################
    # 5. Feature Engineering (Date Calculations)
    # #########################################################################
    reference_date = pd.to_datetime('2021-04-01')

    data_train['days_since_joined'] = (reference_date - data_train['joining_date']) / pd.Timedelta(days=1)
    data_train.drop(columns=['joining_date'], inplace=True)

    data_test['days_since_joined'] = (reference_date - data_test['joining_date']) / pd.Timedelta(days=1)
    data_test.drop(columns=['joining_date'], inplace=True)

    # #########################################################################
    # 6. Complete Case Analysis (CCA) / Row Removal
    # #########################################################################
    data_train = data_train.dropna(how='all')
    data_test = data_test.dropna(how='all')

    data_train = data_train.dropna(subset=['churn_risk_score'])
    data_test = data_test.dropna(subset=['churn_risk_score'])

    # #########################################################################
    # 7. Categorical Imputation
    # #########################################################################
    data_train_mode = data_train['preferred_offer_types'].mode()[0]
    data_train['preferred_offer_types'] = data_train['preferred_offer_types'].fillna(value=data_train_mode)
    data_test['preferred_offer_types'] = data_test['preferred_offer_types'].fillna(value=data_train_mode)

    data_train['region_category'] = data_train['region_category'].fillna(value='Unknown')
    data_test['region_category'] = data_test['region_category'].fillna(value='Unknown')

    # #########################################################################
    # 8. Numerical Imputation (Iterative Imputer / MICE)
    # #########################################################################
    data_train_missing_num = data_train[['days_since_last_login',
                                        'avg_time_spent_in_seconds',
                                        'avg_frequency_login_days',
                                        'points_in_wallet']]

    data_test_missing_num = data_test[['days_since_last_login',
                                        'avg_time_spent_in_seconds',
                                        'avg_frequency_login_days',
                                        'points_in_wallet']]

    imputer = IterativeImputer(max_iter=5, random_state=6)
    imputed_data_train = imputer.fit_transform(data_train_missing_num)

    output_data_train = pd.DataFrame(imputed_data_train,
                                    columns=data_train_missing_num.columns,
                                    index=data_train_missing_num.index)

    data_train[output_data_train.columns] = output_data_train

    imputed_data_test = imputer.transform(data_test_missing_num)

    output_data_test = pd.DataFrame(imputed_data_test,
                                    columns=data_test_missing_num.columns,
                                    index=data_test_missing_num.index)

    data_test[output_data_test.columns] = output_data_test

    # #########################################################################
    # 9. Categorical Encoding (One-Hot Encoding)
    # #########################################################################
    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')

    nominal_columns = ['gender',
                        'region_category',
                        'membership_category',
                        'joined_through_referral',
                        'preferred_offer_types',
                        'medium_of_operation',
                        'internet_option',
                        'used_special_discount',
                        'offer_application_preference',
                        'past_complaint',
                        'complaint_status',
                        'feedback']

    train_ohe = onehot_encoder.fit_transform(data_train[nominal_columns])
    test_ohe = onehot_encoder.transform(data_test[nominal_columns])

    nominal_headers = onehot_encoder.get_feature_names_out(nominal_columns)

    train_ohe_df = pd.DataFrame(data=train_ohe, columns=nominal_headers, index=data_train.index)
    test_ohe_df = pd.DataFrame(data=test_ohe, columns=nominal_headers, index=data_test.index)

    data_train = data_train.drop(columns=nominal_columns).join(train_ohe_df)
    data_test = data_test.drop(columns=nominal_columns).join(test_ohe_df)

    # #########################################################################
    # 10. Separation of Target Variable & Multicollinearity Removal
    # #########################################################################
    feature_cols = [c for c in data_train.columns if c != 'churn_risk_score']

    label_train = data_train['churn_risk_score'].astype('int64')
    label_test = data_test['churn_risk_score'].astype('int64')

    corr_matrix_abs = data_train[feature_cols].corr().abs()

    upper_tri = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))

    threshold = multicoll_corr_threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    data_train = data_train.drop(columns=to_drop)
    data_test = data_test.drop(columns=to_drop)

    # #########################################################################
    # 11. Standardization
    # #########################################################################
    feature_cols = [c for c in data_train.columns if c != 'churn_risk_score']

    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(data_train[feature_cols])
    data_test_scaled = scaler.transform(data_test[feature_cols])

    data_train = pd.DataFrame(data_train_scaled, columns=feature_cols, index=data_train.index)
    data_test = pd.DataFrame(data_test_scaled, columns=feature_cols, index=data_test.index)

    assert data_train.index.equals(label_train.index)
    assert data_test.index.equals(label_test.index)

    data_train = data_train.reset_index(drop=True)
    label_train = label_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    label_test = label_test.reset_index(drop=True)

    label_train = label_train - 1
    label_test = label_test - 1

    return data_train, data_test, label_train, label_test