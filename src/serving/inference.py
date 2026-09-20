import mlflow
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

from pathlib import Path

SERVING_DIR = Path(__file__).resolve().parent

model_artifacts_path = SERVING_DIR / "models" / "m-e10760dbf123478b830c82cd76a5a040" / "artifacts"
model = mlflow.pyfunc.load_model(str(model_artifacts_path))
onehot_enc = joblib.load(SERVING_DIR / "trained_models" / "onehot_encoder.joblib")
standard_scaler = joblib.load(SERVING_DIR / "trained_models" / "standard_scaler.joblib")

model_data = mlflow.models.Model.load(str(model_artifacts_path))

feature_names=[model_data.signature.inputs.to_dict()[i]['name'] for i in range(len(model_data.signature.inputs.to_dict()))]

numeric_features = [
    "age",
    "days_since_last_login",
    "avg_time_spent_in_seconds",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
    "days_since_joined"
]


ui_prediction_feature_fields = [
    "age",                                 #
    "days_since_last_login",               #
    "avg_time_spent_in_seconds",           #
    "avg_transaction_value",               #
    "avg_frequency_login_days",            #
    "points_in_wallet",                    #
    "joining_date",
    "gender",                               #
    "region_category",                       #
    "membership_category",                   #
    "joined_through_referral",               #
    "preferred_offer_types",                #
    "medium_of_operation",                   #
    "internet_option",                      #
    "used_special_discount",                #
    "offer_application_preference",        #
    "past_complaint",                      # 
    "complaint_status",                    #
    "feedback",                             #
]


nominal_columns=['gender', 
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


nominal_columns_ohe = [
    'gender_M',
    'gender_Unknown',
    'region_category_Town',
    'region_category_Unknown',
    'region_category_Village',
    'membership_category_Gold Membership',
    'membership_category_No Membership',
    'membership_category_Platinum Membership',
    'membership_category_Premium Membership',
    'membership_category_Silver Membership',
    'joined_through_referral_No',
    'joined_through_referral_Yes',
    'preferred_offer_types_Gift Vouchers/Coupons',
    'preferred_offer_types_Without Offers',
    'medium_of_operation_Both',
    'medium_of_operation_Desktop',
    'medium_of_operation_Smartphone',
    'internet_option_Mobile_Data',
    'internet_option_Wi-Fi',
    'used_special_discount_Yes',
    'offer_application_preference_Yes',
    'past_complaint_Yes',
    'complaint_status_Solved',
    'complaint_status_Solved in Follow-up',
    'complaint_status_Unsolved',
    'feedback_Poor Customer Service',
    'feedback_Poor Product Quality',
    'feedback_Poor Website',
    'feedback_Products always in Stock',
    'feedback_Quality Customer Care',
    'feedback_Reasonable Price',
    'feedback_Too many ads',
    'feedback_User Friendly Website']


def preprocess(input_df):

    input_df=input_df.copy()

    pars_ohe = onehot_enc.transform(input_df[nominal_columns])
    nominal_headers=onehot_enc.get_feature_names_out(nominal_columns)

    pars_ohe_df=pd.DataFrame(data=pars_ohe, columns=nominal_headers, index=input_df.index)
    input_df=input_df.drop(columns=nominal_columns).join(pars_ohe_df)

    input_df['joining_date']=pd.to_datetime(input_df['joining_date'], errors='coerce')

    reference_date=pd.to_datetime('2021-04-01')
    input_df['days_since_joined']=(reference_date - input_df['joining_date'])/pd.Timedelta(days=1)
    input_df.drop(columns=['joining_date'], inplace=True)

    input_dict=input_df.to_dict(orient='records')[0]
    response_dict=dict.fromkeys(feature_names)

    for i in range(len(numeric_features)):
        response_dict[numeric_features[i]]=input_dict[numeric_features[i]]

    for i in range(len(nominal_columns_ohe)):
        response_dict[nominal_columns_ohe[i]]=input_dict[nominal_columns_ohe[i]]

    response_dict['days_since_joined']=input_dict['days_since_joined']

    response_df_temp=pd.DataFrame([response_dict])
    response_df_scaled=standard_scaler.transform(response_df_temp)

    response_df_temp = pd.DataFrame(response_df_scaled, columns=response_df_temp.columns, index=response_df_temp.index)
    response_df_temp = response_df_temp.reset_index(drop=True)

    response_dict=response_df_temp.to_dict(orient='records')[0]

    return response_dict




def predict(input_dict):
    if hasattr(input_dict, 'model_dump'):
        input_dict = input_dict.model_dump()
        
    elif hasattr(input_dict, 'dict'):
        input_dict = input_dict.dict()

    if isinstance(input_dict, pd.DataFrame):
        input_df = input_dict

    elif isinstance(input_dict, dict):
        if any(isinstance(v, (list, tuple)) for v in input_dict.values()):
            input_df = pd.DataFrame(input_dict)
        else:
            input_df = pd.DataFrame([input_dict])

    elif isinstance(input_dict, list):
        input_df = pd.DataFrame(input_dict)
        
    else:
        input_df = pd.DataFrame(input_dict)

    input_df_enc=preprocess(input_df)

    preds=model.predict(input_df_enc)
    res = preds[0]
    return res.item() if hasattr(res, 'item') else res




if __name__=='__main__':

    input_dict=dict.fromkeys(ui_prediction_feature_fields)

    input_dict["age"]=43
    input_dict["days_since_last_login"]=56
    input_dict["avg_time_spent_in_seconds"]=134
    input_dict["avg_transaction_value"]=45
    input_dict["avg_frequency_login_days"]=6
    input_dict["points_in_wallet"]=45
    input_dict["joining_date"]='2015-04-19'
    input_dict["gender"]='M'
    input_dict["region_category"]='Village'
    input_dict["membership_category"]='Silver Membership'
    input_dict["joined_through_referral"]='Yes'
    input_dict["preferred_offer_types"]='Without Offers'
    input_dict["medium_of_operation"]='Smartphone'
    input_dict["internet_option"]='Mobile_Data'
    input_dict["used_special_discount"]='Yes'
    input_dict["offer_application_preference"]='Yes'
    input_dict["past_complaint"]='No'
    input_dict["complaint_status"]='Not Applicable'
    input_dict["feedback"]='Poor Website'

    print(predict(input_dict))




