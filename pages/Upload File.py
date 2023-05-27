import streamlit as st
import pandas as pd
from io import StringIO
import pymysql
import pickle
from datetime import datetime

# Thiết lập kết nối
connection = pymysql.connect(
    host="127.0.0.1",
    user="root",
    password="taimysql1001",
    database="telco_churn"
)
def clean(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop(['total_day_minutes','total_night_minutes','total_eve_minutes','total_intl_minutes'],axis = 1)
    return df

def fe(df):
    data_predict = df.filter(items=['account_length', 'international_plan', 'voice_mail_plan',
                                           'number_vmail_messages', 'total_day_calls', 'total_day_charge',
                                           'total_eve_calls', 'total_eve_charge', 'total_night_calls',
                                           'total_night_charge', 'total_intl_calls', 'total_intl_charge',
                                           'number_customer_service_calls'])
    data_predict['international_plan'] = data_predict['international_plan'].map({"no": 0, "yes": 1})
    data_predict['voice_mail_plan'] = data_predict['voice_mail_plan'].map({"no": 0, "yes": 1})
    return  data_predict
uploaded_file = st.file_uploader("Upload File")
if uploaded_file is not None:
    data_raw = pd.read_csv(uploaded_file)
    data_raw = clean(data_raw)
    st.write(data_raw)
    data_predict = fe(data_raw)
    # st.write(data_predict)
    model = pickle.load(open('Model/churn_model.sav', 'rb'))
    y = model.predict(data_predict)
    data_raw['status'] = y
    mapping = {0: 'non_churn', 1: 'churn'}
    data_raw['status'] = data_raw['status'].map(mapping)

    # Đẩy giữ liệu vào dim_date
    now = datetime.now()
    day = now.day
    month = now.strftime("%m")
    year = now.year
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    second = now.strftime("%S")
    date_id = f'{year}{month}{day}{hour}{minute}{second}'
    cursor = connection.cursor()
    insert_query = "INSERT INTO telco_churn.dim_date(date_id, record_date) VALUES (%s, %s)"
    cursor.execute(insert_query, (date_id, now))
    connection.commit()
    cursor.close()

    # Đẩy giữ liệu vào dim_customer
    dataframe = data_raw.drop('status', axis=1)
    with connection.cursor() as cursor:
        for _, row in dataframe.iterrows():
            insert_query = f"INSERT INTO telco_churn.dim_customer (phone,state,account_length,international_plan,voice_mail_plan,number_vmail_messages,total_day_calls,total_day_charge,total_eve_calls,total_eve_charge,total_night_calls,total_night_charge,total_intl_calls,total_intl_charge,number_customer_service_calls) VALUES (%s, %s, %s, %s, %s,%s, %s, %s, %s, %s,%s, %s, %s, %s, %s)"
            cursor.execute(insert_query, tuple(row))
    connection.commit()

    # Đẩy giữ liệu vào fact_churn_analysis
    status = data_raw['status']
    phone = data_raw['phone']
    with connection.cursor() as cursor:
        for value, phone_value in zip(status, phone):
            insert_query1 = "INSERT INTO telco_churn.fact_churn_analysis (date_id, phone, status) VALUES (%s, %s, %s)"
            cursor.execute(insert_query1, (date_id, phone_value, value))
        connection.commit()

    # # Đóng kết nối
    connection.close()

    st.success("Data has been pushed to database.")
