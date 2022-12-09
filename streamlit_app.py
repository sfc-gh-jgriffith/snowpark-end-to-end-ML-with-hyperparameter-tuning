import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.functions import call_udf, col
import json
import pydeck as pdk
import altair as alt

# connect to snowflake
with open('creds.json') as f:
    connection_parameters = json.load(f)

session = Session.builder.configs(connection_parameters).create()

def custom_round(x, base=5):
    return int(round(float(x)/base)) * base

@st.cache
def get_data():
    test_data = session.table("MEMBERSHIP_TEST").sample(.25).to_pandas()
    

    test_data["YEARLY_SPENT_BIN"] = test_data["YEARLY_SPENT"].apply(lambda x: custom_round(x, 5))

    return test_data

test_data = get_data()


features = ['GENDER', 'MEMBERSHIP_STATUS', 'MEMBERSHIP_LENGTH',
                    'AVG_SESSION_LENGTH', 'TIME_ON_APP', 'TIME_ON_WEBSITE']
                    

binned_df = pd.DataFrame(test_data.groupby(["YEARLY_SPENT_BIN"]).size())
binned_df.columns = ["COUNT"]

def update_handler(model_results):
    binned_df["OTHER"] = binned_df["COUNT"]
    binned_df["PREDICTED"] = 0

    model_results_bin = custom_round(model_results, 5)
    binned_df.loc[binned_df.index == model_results_bin,"PREDICTED"] = binned_df["COUNT"]
    binned_df.loc[binned_df.index == model_results_bin,"OTHER"] = 0

    return binned_df




st.title("MEMBERSHIP LTV MODEL EXPLORER")
st.markdown("""Interacting with the model hosted on Snowflake. Each change in the inputs makes a call to Snowflake and gets the model score. 
The model score is then plotted against a sample of the source table from Snowflake.""")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", options=test_data['GENDER'].drop_duplicates())
    MEMBERSHIP_STATUS = st.selectbox("Membership Status", options=test_data['MEMBERSHIP_STATUS'].drop_duplicates())
with col2:
    membership_length = st.slider("Membership Length", value=test_data['MEMBERSHIP_LENGTH'].median(), min_value=test_data['MEMBERSHIP_LENGTH'].min(), max_value=test_data['MEMBERSHIP_LENGTH'].max())
    avg_session_length = st.slider("Average Session Length", value=test_data['AVG_SESSION_LENGTH'].median(), min_value=test_data['AVG_SESSION_LENGTH'].min(), max_value=test_data['AVG_SESSION_LENGTH'].max())
with col3:
    time_on_app = st.slider("Time on App", value=test_data['TIME_ON_APP'].median(), min_value=test_data['TIME_ON_APP'].min(), max_value=test_data['TIME_ON_APP'].max())
    time_on_website = st.slider("Time on Website", value=test_data['TIME_ON_WEBSITE'].median(), min_value=test_data['TIME_ON_WEBSITE'].min(), max_value=test_data['TIME_ON_WEBSITE'].max())




from sklearn.preprocessing import OneHotEncoder

model_inputs = pd.DataFrame([[gender, MEMBERSHIP_STATUS, membership_length, avg_session_length, time_on_app, time_on_website]], columns=features)

model_inputs["MEMBERSHIP_STATUS_BASIC"] = (model_inputs["MEMBERSHIP_STATUS"] == "BASIC").astype(int)
model_inputs["MEMBERSHIP_STATUS_BRONZE"] = (model_inputs["MEMBERSHIP_STATUS"] == "BRONZE").astype(int)
model_inputs["MEMBERSHIP_STATUS_DIAMOND"] = (model_inputs["MEMBERSHIP_STATUS"] == "DIAMOND").astype(int)
model_inputs["MEMBERSHIP_STATUS_GOLD"] = (model_inputs["MEMBERSHIP_STATUS"] == "GOLD").astype(int)
model_inputs["MEMBERSHIP_STATUS_PLATIN"] = (model_inputs["MEMBERSHIP_STATUS"] == "PLATIN").astype(int)
model_inputs["MEMBERSHIP_STATUS_SILVER"] = (model_inputs["MEMBERSHIP_STATUS"] == "SILVER").astype(int)


model_inputs["GENDER_FEMALE"] = (model_inputs["GENDER"] == "FEMALE").astype(int)
model_inputs["GENDER_MALE"] = (model_inputs["GENDER"] == "MALE").astype(int)
model_inputs["GENDER_UNKNOWN"] = (model_inputs["GENDER"] == "UNKNOWN").astype(int)

model_features = ['MEMBERSHIP_LENGTH',
                 'AVG_SESSION_LENGTH',
                 'TIME_ON_APP',
                 'TIME_ON_WEBSITE',
                 'GENDER_FEMALE',
                 'GENDER_MALE',
                 'GENDER_UNKNOWN',
                 'MEMBERSHIP_STATUS_BASIC',
                 'MEMBERSHIP_STATUS_BRONZE',
                 'MEMBERSHIP_STATUS_DIAMOND',
                 'MEMBERSHIP_STATUS_GOLD',
                 'MEMBERSHIP_STATUS_PLATIN',
                 'MEMBERSHIP_STATUS_SILVER']

model_inputs = model_inputs[model_features]

# sending raw values performs better than creating an intermediate snowpark dataframe off of pandas
#model_results = session.create_dataframe(model_inputs, schema=model_features).select(call_udf("predict", *[col(c) for c in model_features]).alias('PREDICTION')).to_pandas().iloc[0]['PREDICTION']
model_results = session.sql(f"select predict({','.join([str(v) for v in model_inputs.values[0]])}) as prediction").to_pandas().iloc[0]["PREDICTION"]


model_results_text = st.empty()
chart = st.empty()

pandas_model_inputs = pd.DataFrame([[gender, MEMBERSHIP_STATUS, membership_length, avg_session_length, time_on_app, time_on_website, model_results, "YES"]], columns=features+["YEARLY_SPENT","PREDICTION"])

chart_data = pd.concat([test_data.copy(), pandas_model_inputs], axis=0)
chart_data["PREDICTION"].fillna("NO", inplace=True)

chart_data = chart_data[chart_data["GENDER"] == gender]
chart_data = chart_data[chart_data["MEMBERSHIP_STATUS"] == MEMBERSHIP_STATUS]

chart_col1, chart_col2 = st.columns(2)

counter = 0
for col in ['MEMBERSHIP_LENGTH', 'AVG_SESSION_LENGTH', 'TIME_ON_APP', 'TIME_ON_WEBSITE']:
    c = alt.Chart(chart_data).mark_circle().encode(x=col, 
                                                  y='YEARLY_SPENT', 
                                                  color=alt.Color('PREDICTION', legend=None), 
                                                  tooltip=['MEMBERSHIP_LENGTH', 'YEARLY_SPENT', 'YEARLY_SPENT'])
    if counter % 2 == 0:
         chart_col1.altair_chart(c)
    else:
         chart_col2.altair_chart(c)
    counter += 1



model_results_text.markdown(f"## Model Estimate: {model_results.round(2)}")

chart.bar_chart(update_handler(model_results)[["PREDICTED","OTHER"]], height=250)





