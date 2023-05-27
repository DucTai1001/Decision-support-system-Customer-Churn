import numpy as np
import pandas as pd
import streamlit as st
import cufflinks as cf
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
st.set_page_config(layout="wide")
from plotly.subplots import make_subplots
#Khai báo kết nối DB
conn = st.experimental_connection(
    "local_db",
    type="sql",
    url="mysql://root:taimysql1001@127.0.0.1:3306/telco_churn"
)
tab0, tab1 = st.tabs(["Overview", "Payment&Service"])
#Tab0
with tab0:
    #Khai báo thống kê
    tab0.subheader("Overview")
    num_churn = conn.query("select count(phone) from telco_churn.fact_churn_analysis where status = 'churn' ")
    num_churn = num_churn.iloc[0, 0]
    custom_num = conn.query("select count(date_id) from telco_churn.fact_churn_analysis")
    custom_num = custom_num.iloc[0, 0]
    col001, col002 = st.columns(2)
    with col001:
        st.metric(label="Number of Customers", value=custom_num)
    with col002:
        st.metric(label="Number of potient churn", value=num_churn)
    container00 = st.container()
    col011, col012 = st.columns(([6,5.5]))

    with container00:
        with col011:
            # Biểu đồ churn rates
            churn_rate = conn.query("select phone, status from telco_churn.fact_churn_analysis")
            churn_rate = churn_rate.groupby(by=["status"]).count()[['phone']].rename(columns={"phone": "Count"}).reset_index()
            colors = ['#FF5F5F', '#5FBA7D']
            churn_rate_pie_fig = churn_rate.iplot(
                kind="pie",
                labels="status",
                values="Count",
                title="Potential churn rates",
                asFigure=True,
                hole=0.4,
            )
            churn_rate_pie_fig.update_layout(
                title={
                    'text': "Potential churn rates",
                    'font': {
                        'size': 24  # Đặt kích thước chữ là 24
                    }
                }
            )
            churn_rate_pie_fig.update_layout(
                legend=dict(
                    x=0.5,  # Vị trí ngang của chú thích (0.5 tương ứng với giữa)
                    y=-0.1,  # Vị trí dọc của chú thích (-0.1 tương ứng với phía dưới)
                    orientation='h'  # Định dạng chú thích là ngang (horizontal)
                )
            )
            churn_rate_pie_fig.update_layout(
                width=510,
                height=400
            )
            churn_rate_pie_fig
        with col012:
            # Biểu đồ features important
            model = pickle.load(open('Model/churn_model.sav', 'rb'))
            importance = model.feature_importances_
            feature_names = ['account length', 'international plan', 'voice mail_plan',
                             'number vmail messages', 'total day minutes', 'total day calls',
                             'total day charge', 'total eve minutes', 'total eve calls',
                             'total eve charge', 'total night minutes', 'total night calls',
                             'total night charge', 'total intl minutes', 'total intl calls',
                             'total intl charge', 'number customer service calls']
            sorted_idx = sorted(range(len(importance)), key=lambda k: importance[k], reverse=False)
            sorted_feature_names = [feature_names[i] for i in sorted_idx]
            sorted_importance = [importance[i] for i in sorted_idx]
            data = pd.DataFrame({'Feature': sorted_feature_names, 'Importance': sorted_importance})

            feature_fig = data.iplot(kind='barh', x='Feature', y='Importance',title = "Features important", asFigure=True)
            feature_fig.update_layout(
                title={
                    'text': "Features important",
                    'font': {
                        'size': 24  # Đặt kích thước chữ là 24
                    }
                }
            )
            feature_fig.update_layout(
                width=650,  # Đặt giá trị chiều rộng của Figure (đơn vị: pixel)
                height=400  # Đặt giá trị chiều cao của Figure (đơn vị: pixel)
            )
            feature_fig

#Tab charge&Service
with tab1:
    container2 = st.container()
    col101, col102 = st.columns(([6,5.5]))
    df = conn.query('select c.phone, state ,account_length, international_plan, voice_mail_plan,number_vmail_messages, total_day_calls, total_day_charge,total_eve_calls, total_eve_charge, total_night_calls,total_night_charge, total_intl_calls, total_intl_charge,number_customer_service_calls, a.status from dim_customer as c, fact_churn_analysis as a where a.phone = c.phone')
    with container2:
        with col101:
            df_intpl_yes = df[df["international_plan"] == "yes"].groupby(by=["status"]).count()[['phone']].rename(
                columns={"phone": "Count"}).reset_index()
            df_intpl_no = df[df["international_plan"] == "no"].groupby(by=["status"]).count()[['phone']].rename(
                columns={"phone": "Count"}).reset_index()
            fig_intpl = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]], horizontal_spacing=0.0000005)
            fig_intpl.add_trace(go.Pie(
                labels=df_intpl_yes['status'],
                values=df_intpl_yes['Count'],
                title='International Plan: Yes',
                marker=dict(colors=['#3780bf', '#ff9933'])
            ), row=1, col=1)
            fig_intpl.add_trace(go.Pie(
                labels=df_intpl_no['status'],
                values=df_intpl_no['Count'],
                title='International Plan: No',
                marker=dict(colors=['#3780bf', '#ff9933'])
            ), row=1, col=2)
            fig_intpl.update_traces(textfont_size=12)
            fig_intpl.update_layout(title='Churn proportion by International Plan', title_x=0.2, grid=dict(rows=1, columns=2))
            fig_intpl.update_layout(width=450, height=350)
            fig_intpl.update_layout(
                legend=dict(
                    x=0.5,  # Vị trí ngang của chú thích (0.5 tương ứng với giữa)
                    y=-0.1,  # Vị trí dọc của chú thích (-0.1 tương ứng với phía dưới)
                    orientation='h'  # Định dạng chú thích là ngang (horizontal)
                )
            )
            fig_intpl.update_traces(textfont_size=8)
            fig_intpl
        with col102:
            # df_vmpl_yes = df[df["voice_mail_plan"] == "yes"].groupby(by=["status"]).count()[['phone']].rename(
            #     columns={"phone": "Count"}).reset_index()
            # df_vmpl_no = df[df["voice_mail_plan"] == "no"].groupby(by=["status"]).count()[['phone']].rename(
            #     columns={"phone": "Count"}).reset_index()
            # fig_vmpl = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
            #                           horizontal_spacing=0.0000005)
            # fig_vmpl.add_trace(go.Pie(
            #     labels=df_vmpl_yes['status'],
            #     values=df_vmpl_yes['Count'],
            #     title='Voice Mail Plan: Yes',
            #     marker=dict(colors=['#3780bf', '#ff9933'])
            # ), row=1, col=1)
            # fig_vmpl.add_trace(go.Pie(
            #     labels=df_vmpl_no['status'],
            #     values=df_vmpl_no['Count'],
            #     title='Voice Mail Plan: No',
            #     marker=dict(colors=['#3780bf', '#ff9933'])
            # ), row=1, col=2)
            # fig_vmpl.update_traces(textfont_size=12)
            # fig_vmpl.update_layout(title='Churn proportion by Voice Mail Plan', title_x=0.2,
            #                         grid=dict(rows=1, columns=2))
            # fig_vmpl.update_layout(width=450, height=350)
            # fig_vmpl.update_layout(
            #     legend=dict(
            #         x=0.5,  # Vị trí ngang của chú thích (0.5 tương ứng với giữa)
            #         y=-0.1,  # Vị trí dọc của chú thích (-0.1 tương ứng với phía dưới)
            #         orientation='h'  # Định dạng chú thích là ngang (horizontal)
            #     )
            # )
            # fig_vmpl.update_traces(textfont_size=8)
            # fig_vmpl

            df_state_churn = df[df["status"] == "churn"].groupby(by=["state"]).count()[['phone']].rename(
                columns={"phone": "Count"}).reset_index()
            df_state_nchurn = df[df["status"] == "non_churn"].groupby(by=["state"]).count()[['phone']].rename(
                columns={"phone": "Count"}).reset_index()

            # Plotting the 'df_state_churn' DataFrame
            fig_state = go.Figure(data=go.Bar(x=df_state_churn['state'], y=df_state_churn['Count'], name='Churn'))
            fig_state.update_layout(title='Churn by State', xaxis_title='State', yaxis_title='Count')

            # Plotting the 'df_state_nchurn' DataFrame
            fig_state.add_trace(
                go.Bar(x=df_state_nchurn['state'], y=df_state_nchurn['Count'], name='Non-Churn', opacity=0.5))
            fig_state

    container3 = st.container()
    with container3:
        import plotly.graph_objects as go

        # Create boxplot traces for churn and non-churn categories for each column
        trace_day_churn = go.Box(
            y=df[df['status'] == 'churn']['total_day_charge'],
            name='Churn - Total Day Charge'
        )

        trace_day_non_churn = go.Box(
            y=df[df['status'] == 'non_churn']['total_day_charge'],
            name='Non-Churn - Total Day Charge'
        )

        trace_night_churn = go.Box(
            y=df[df['status'] == 'churn']['total_night_charge'],
            name='Churn - Total Night Charge'
        )

        trace_night_non_churn = go.Box(
            y=df[df['status'] == 'non_churn']['total_night_charge'],
            name='Non-Churn - Total Night Charge'
        )

        trace_eve_churn = go.Box(
            y=df[df['status'] == 'churn']['total_eve_charge'],
            name='Churn - Total Eve Charge'
        )

        trace_eve_non_churn = go.Box(
            y=df[df['status'] == 'non_churn']['total_eve_charge'],
            name='Non-Churn - Total Eve Charge'
        )

        trace_intl_churn = go.Box(
            y=df[df['status'] == 'churn']['total_intl_charge'],
            name='Churn - Total Intl Charge'
        )

        trace_intl_non_churn = go.Box(
            y=df[df['status'] == 'non_churn']['total_intl_charge'],
            name='Non-Churn - Total Intl Charge'
        )

        # Create a figure and add the traces to the figure
        fig = go.Figure(data=[
            trace_day_churn, trace_day_non_churn,
            trace_night_churn, trace_night_non_churn,
            trace_eve_churn, trace_eve_non_churn,
            trace_intl_churn, trace_intl_non_churn
        ])

        # Configure the layout
        fig.update_layout(
            title='Boxplots of Charges (Churn vs. Non-Churn)',
            yaxis=dict(title='Charge')
        )
        fig.update_layout(
            width=1200,
            height=400
        )
        fig

        container4 = st.container()
        with container4:
            import plotly.graph_objects as go

            lenght_churn = df[df["status"] == "churn"].groupby("state")["account_length"].mean().reset_index()
            lenght_nchurn = df[df["status"] == "non_churn"].groupby("state")["account_length"].mean().reset_index()

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=lenght_churn['state'], y=lenght_churn['account_length'],
                                     mode='lines', name='Churn Customers',
                                     line=dict(color='red')))
            fig.add_trace(go.Scatter(x=lenght_nchurn['state'], y=lenght_nchurn['account_length'],
                                     mode='lines', name='Non-Churn Customers'))

            fig.update_layout(title='Account Length by State',
                              xaxis_title='State',
                              yaxis_title='Account Length')

            fig





