import numpy as np
import pandas as pd
import streamlit as st
import cufflinks as cf
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go

conn = st.experimental_connection(
    "local_db",
    type="sql",
    url="mysql://root:taimysql1001@127.0.0.1:3306/telco_churn"
)

df = conn.query(
    'select c.phone,state, account_length, international_plan, voice_mail_plan,number_vmail_messages, total_day_calls, total_day_charge,total_eve_calls, total_eve_charge, total_night_calls,total_night_charge, total_intl_calls, total_intl_charge,number_customer_service_calls, a.status from dim_customer as c, fact_churn_analysis as a where a.phone = c.phone')


status = df['status'].unique().tolist()
status_select = st.multiselect('Status:', status, default=status)
st.write(df[df['status'].isin(status_select)])