from pydantic import BaseModel
from xgboost import XGBClassifier
import sklearn
import pandas as pd
from fastapi import FastAPI
import joblib


# Defining input class
class HomeCreditRisk(BaseModel):
    CODE_GENDER: float
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_EDUCATION_TYPE: float
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    OWN_CAR_AGE: float
    FLAG_EMP_PHONE: float
    REGION_RATING_CLIENT: float
    HOUR_APPR_PROCESS_START: float
    REG_CITY_NOT_LIVE_CITY: float
    REG_CITY_NOT_WORK_CITY: float
    LIVE_CITY_NOT_WORK_CITY: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    APARTMENTS_AVG: float
    YEARS_BEGINEXPLUATATION_AVG: float
    YEARS_BUILD_AVG: float
    ELEVATORS_AVG: float
    FLOORSMAX_AVG: float
    FLOORSMIN_AVG: float
    LANDAREA_AVG: float
    LIVINGAREA_AVG: float
    APARTMENTS_MODE: float
    BASEMENTAREA_MODE: float
    YEARS_BEGINEXPLUATATION_MODE: float
    YEARS_BUILD_MODE: float
    ELEVATORS_MODE: float
    ENTRANCES_MODE: float
    FLOORSMAX_MODE: float
    LANDAREA_MODE: float
    LIVINGAPARTMENTS_MODE: float
    LIVINGAREA_MODE: float
    APARTMENTS_MEDI: float
    BASEMENTAREA_MEDI: float
    YEARS_BUILD_MEDI: float
    FLOORSMAX_MEDI: float
    LANDAREA_MEDI: float
    LIVINGAREA_MEDI: float
    TOTALAREA_MODE: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_3: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    NAME_INCOME_TYPE_Pensioner: float
    NAME_FAMILY_STATUS_Married: float
    OCCUPATION_TYPE_Laborers: float
    FONDKAPREMONT_MODE_XNA: float
    HOUSETYPE_MODE_block_of_flats: float
    HOUSETYPE_MODE_XNA: float
    EMERGENCYSTATE_MODE_No: float
    EMERGENCYSTATE_MODE_XNA: float
    RATIO_ANNUITY_TO_INCOME: float
    REGION_TO_INCOME: float
    RATIO_CREDIT_TO_EXT_SOURCE: float
    SUM_EXT_SOURCES: float
    MAX_EXTERNAL_SOURCES: float
    MIN_EXT_SOURCES: float
    PROD_EXT_SOURCES: float
    CREDIT_ANNUITY_RATIO: float
    PROD_REGION_POPULATION_AMT_CREDIT: float
    PROD_REGION_RATING_AMT_INCOME: float
    INCOME_PER_CHILD: float
    INCOME_PER_PERSON: float
    PAYMENT_RATE: float
    MEAN_50_NN: float
    FRESH_MONTHS_BALANCE_MEAN_MIN_ACTIVE: float
    FRESH_MONTHS_BALANCE_MEAN_MAX_ACTIVE: float
    CREDIT_COUNT_ACTIVE: float
    CREDIT_COUNT_CLOSED: float
    FRESH_MONTHS_BALANCE_MIN: float
    FRESH_SK_DPD_DEF_SUM: float
    OLD_MONTHS_BALANCE_MAX: float
    OLD_CNT_INSTALMENT_MEAN: float
    OLD_CNT_INSTALMENT_SUM: float
    OLD_CNT_INSTALMENT_FUTURE_MEAN: float
    OLD_SK_DPD_COUNT: float
    OLD_SK_DPD_DEF_COUNT: float
    NAME_CONTRACT_STATUS_ACTIVE: float
    NAME_CONTRACT_STATUS_COMPLETED: float
    FRESH_NUM_INSTALMENT_VERSION_MIN: float
    FRESH_NUM_INSTALMENT_NUMBER_MAX: float
    FRESH_AMT_INSTALMENT_MIN: float
    FRESH_AMT_PAYMENT_MIN: float
    FRESH_DIFF_DAYS_PAYMNET_MIN: float
    OLD_NUM_INSTALMENT_VERSION_SUM: float
    OLD_NUM_INSTALMENT_VERSION_MIN: float
    OLD_NUM_INSTALMENT_NUMBER_MEAN: float
    OLD_NUM_INSTALMENT_NUMBER_MAX: float
    OLD_NUM_INSTALMENT_NUMBER_SUM: float
    OLD_NUM_INSTALMENT_NUMBER_MIN: float
    OLD_DAYS_INSTALMENT_MEAN: float
    OLD_DAYS_INSTALMENT_MAX: float
    OLD_DAYS_INSTALMENT_SUM: float
    OLD_DAYS_INSTALMENT_MIN: float
    OLD_DAYS_ENTRY_PAYMENT_MAX: float
    OLD_DAYS_ENTRY_PAYMENT_SUM: float
    OLD_DAYS_ENTRY_PAYMENT_MIN: float
    OLD_AMT_INSTALMENT_MEAN: float
    OLD_AMT_INSTALMENT_MAX: float
    OLD_AMT_INSTALMENT_SUM: float
    OLD_AMT_INSTALMENT_MIN: float
    OLD_AMT_PAYMENT_MEAN: float
    OLD_AMT_PAYMENT_MAX: float
    OLD_AMT_PAYMENT_SUM: float
    OLD_AMT_PAYMENT_MIN: float
    OLD_DIFF_DAYS_PAYMNET_MEAN: float
    OLD_DIFF_DAYS_PAYMNET_SUM: float
    OLD_DIFF_DAYS_PAYMNET_MIN: float
    OLD_RATIO_AMT_PAYMENT_MEAN: float
    OLD_RATIO_AMT_PAYMENT_MAX: float
    OLD_RATIO_AMT_PAYMENT_SUM: float
    CNT_PAYMENT_MIN: float
    CNT_PAYMENT_MAX: float
    CNT_PAYMENT_MEAN: float
    HOUR_APPR_PROCESS_START_MIN: float
    DAYS_DECISION_MAX: float
    SELLERPLACE_AREA_MAX: float
    SK_ID_PREV_COUNT: float
    PRODUCT_COMBINATION_RARE_SUM: float
    PRODUCT_COMBINATION_CARD_STREET_SUM: float
    PRODUCT_COMBINATION_CARD_X_SELL_SUM: float
    NAME_YIELD_GROUP_XNA_SUM: float
    NAME_YIELD_GROUP_MIDDLE_SUM: float
    NAME_CONTRACT_STATUS_REFUSED_SUM: float
    CODE_REJECT_REASON_XAP_SUM: float
    CODE_REJECT_REASON_LIMIT_SUM: float
    CODE_REJECT_REASON_HC_SUM: float
    CODE_REJECT_REASON_SCOFR_SUM: float
    CHANNEL_TYPE_CREDIT_AND_CASH_OFFICES_SUM: float
    CHANNEL_TYPE_AP___CASH_LOAN__SUM: float
    NAME_PRODUCT_TYPE_XNA_SUM: float
    NAME_PRODUCT_TYPE_WALK_IN_SUM: float
    NAME_GOODS_CATEGORY_XNA_SUM: float
    NAME_CASH_LOAN_PURPOSE_XNA_SUM: float
    NAME_CASH_LOAN_PURPOSE_XAP_SUM: float
    NAME_CASH_LOAN_PURPOSE_RARE_SUM: float
    NAME_PORTFOLIO_XNA_SUM: float
    NAME_PORTFOLIO_CASH_SUM: float
    NAME_PORTFOLIO_CARDS_SUM: float
    NAME_CONTRACT_TYPE_REVOLVING_LOANS_SUM: float
    NAME_PAYMENT_TYPE_CASH_THROUGH_THE_BANK_SUM: float
    NAME_CLIENT_TYPE_REPEATER_SUM: float
    NAME_TYPE_SUITE_XNA_SUM: float
    NAME_TYPE_SUITE_UNACCOMPANIED_SUM: float
    WEEKDAY_APPR_PROCESS_START_FRIDAY_SUM: float
    WEEKDAY_APPR_PROCESS_START_THURSDAY_SUM: float
    FRESH_MONTHS_BALANCE_MEAN: float
    FRESH_AMT_CREDIT_LIMIT_ACTUAL_MIN: float
    FRESH_AMT_CREDIT_LIMIT_ACTUAL_SUM: float
    FRESH_AMT_DRAWINGS_ATM_CURRENT_MAX: float
    FRESH_AMT_DRAWINGS_ATM_CURRENT_SUM: float
    FRESH_AMT_PAYMENT_CURRENT_MIN: float
    FRESH_CNT_DRAWINGS_ATM_CURRENT_MEAN: float
    FRESH_CNT_DRAWINGS_ATM_CURRENT_MAX: float
    FRESH_CNT_DRAWINGS_ATM_CURRENT_SUM: float
    FRESH_CNT_DRAWINGS_CURRENT_MAX: float
    FRESH_CNT_INSTALMENT_MATURE_CUM_MAX: float
    FRESH_CNT_INSTALMENT_MATURE_CUM_SUM: float
    OLD_MONTHS_BALANCE_SUM: float
    OLD_AMT_BALANCE_MEAN: float
    OLD_AMT_CREDIT_LIMIT_ACTUAL_MAX: float
    OLD_AMT_CREDIT_LIMIT_ACTUAL_SUM: float
    OLD_AMT_DRAWINGS_ATM_CURRENT_MEAN: float
    OLD_AMT_INST_MIN_REGULARITY_MAX: float
    OLD_AMT_PAYMENT_CURRENT_SUM: float
    OLD_AMT_RECEIVABLE_PRINCIPAL_SUM: float
    OLD_CNT_DRAWINGS_ATM_CURRENT_MEAN: float
    OLD_CNT_DRAWINGS_ATM_CURRENT_SUM: float
    OLD_CNT_DRAWINGS_CURRENT_MEAN: float
    OLD_CNT_DRAWINGS_CURRENT_MAX: float
    OLD_SK_DPD_MAX: float
    OLD_SK_DPD_DEF_SUM: float
    OLD_BALANCE_LIMIT_RATIO_MAX: float
    OLD_AMT_DRAWING_ALL_MEAN: float
    OLD_AMT_DRAWING_ALL_MAX: float


# Defining output class
class Predictions(BaseModel):
    default_proba: float


# =========================================================

# Starting the app
app = FastAPI()

# Loading the model
model = joblib.load("model.joblib")


# Initialising home page
@app.get("/")
def home_page():
    return {"message": "Home Credit Default Risk App", "model_version": 0.1}


# Inferencing endpoint
@app.post("/predict", response_model=Predictions)
def predict(data: HomeCreditRisk):
    # Creating dataframe
    df = pd.DataFrame([data.dict()])
    # Make predictions
    pred = model.predict_proba(df)[0, 1]
    return {"default_proba": pred}
