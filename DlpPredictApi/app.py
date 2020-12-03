import json
import joblib
import pandas as pd
import numpy as np

# List of all the features present in the model
FEATURE_LIST = [
    'AGE',
    'CREDIT_CARD_NUMBER',
    'DATE_OF_BIRTH,',
    'DISABILITY_GROUP',
    'EDUCATION_LEVEL',
    'EMAIL_ADDRESS',
    'FEMALE_NAME,',
    'FIRST_NAME',
    'GENDER',
    'GENERIC_ID',
    'HDX_HEADERS',
    'HH_ATTRIBUTES,',
    'ICD10_CODE',
    'ICD9_CODE',
    'IMEI_HARDWARE_ID',
    'IP_ADDRESS,',
    'LAST_NAME',
    'LOCATION',
    'MALE_NAME',
    'MARITAL_STATUS',
    'MEDICAL_TERM,',
    'OCCUPATION',
    'ORGANIZATION_NAME',
    'PERSON_NAME',
    'PHONE_NUMBER,',
    'PROTECTION_GROUP',
    'RELIGIOUS_GROUPS',
    'SPOKEN_LANGUAGE,',
    'STREET_ADDRESS',
    'TIME',
    'URL'
    ]

def predict(file_url):
    """Returns a prediction."""
    with open(file_url, encoding='utf-8') as f:
        data = json.load(f)
    df = pd.json_normalize(data['dlp_scan_results'])
    df = df.drop_duplicates()
    df['likelihood_new'] = df.likelihood
    df = df.replace({'likelihood_new': r'POSSIBLE'}, {'likelihood_new': 0.33})
    df = df.replace({'likelihood_new': r'LIKELY'}, {'likelihood_new': 0.66})
    df = df.replace({'likelihood_new': r'VERY_LIKELY'}, {'likelihood_new': 0.95})
    df = df.pivot_table(columns='infoType', values='likelihood_new', aggfunc="max")
    data_features = df.iloc[0]
    features = np.zeros((1, len(FEATURE_LIST)))
    index = 0
    while (index < len(FEATURE_LIST)):
        if FEATURE_LIST[index] in data_features:
            features[0, index] = data_features[FEATURE_LIST[index]]
        index = index + 1

    rf_model = joblib.load(r'models/rf_model.joblib')
    rf_score = rf_model.predict(features)

    gb_model = joblib.load(r'models/gb_model.joblib')
    gb_score = gb_model.predict(features)

    gl_model = joblib.load(r'models/gl_model.joblib')
    gl_score = gl_model.predict(features)
       
    total_score = rf_score[0] + gb_score[0] + gl_score[0]
    average_score = total_score/3
    return average_score
