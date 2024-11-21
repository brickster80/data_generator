# app.py
import streamlit as st
from langchain import PromptTemplate
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import ast
#from pyspark.sql import SparkSession
import re
from streamlit.web.server.websocket_headers import _get_websocket_headers
from databricks import sql
import pandas as pd
# connect to warehouse
from databricks.connect.session import DatabricksSession
from databricks import sql
from databricks.sdk.core import Config
import os
import numpy as np
from faker import Faker

#connect to sql
cfg = Config()

# Set page configuration with custom favicon
st.set_page_config(
    page_title="DemoArigato",
    page_icon="static/drobot.png"  # Replace with the path to your favicon file
)
left_co, cent_co,last_co = st.columns(3)

def _get_user_info():
    headers = st.context.headers #_get_websocket_headers()
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
        access_token=headers.get("X-Forwarded-Access-Token")
    )

user_info = _get_user_info()

# Initialize Databricks workspace client
w = WorkspaceClient()

brickthrough = 'http://go/brickthroughs-round1'
#st.toast("Thanks all for voting for DemoArigato!")

with cent_co:
    st.image("static/drobot.png")

#header {visibility: hidden;}
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stApp header {
                background-color: #f9f7f4;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#fake data generator
fake = Faker()
Faker.seed(42)


# Function to call the model endpoint
def call_model_endpoint(endpoint_name, messages, max_tokens=512, timeout_minutes=10):
    chat_messages = [
        ChatMessage(
            content=message["content"],
            role=ChatMessageRole[message["role"].upper()]
        ) if isinstance(message, dict) else ChatMessage(content=message, role=ChatMessageRole.USER)
        for message in messages
    ]
    response = w.serving_endpoints.query(
        name=endpoint_name,
        messages=chat_messages,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
  
# Function to run the chain
def run_chain(prompt_template, **kwargs):
    formatted_prompt = prompt_template.format(**kwargs)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": formatted_prompt}
    ]
    response = call_model_endpoint("databricks-meta-llama-3-1-405b-instruct", messages)
    return response

# query and save synthetic data
def sqlQuery(query: str) -> pd.DataFrame:
    # ensure the right environment variables are set
    def defined(var: str) -> bool: return os.getenv(var) is not None
    assert defined('DATABRICKS_WAREHOUSE_ID') and os.getenv('DATABRICKS_WAREHOUSE_ID') != "<your warehouse ID>", "To use SQL, set DATABRICKS_WAREHOUSE_ID in app.yaml. You can find your SQL Warehouse ID by navigating to SQL Warehouses, clicking on your warehouse, and then looking for the ID next to the Name."
    assert defined('DATABRICKS_HOST'), "To run outside of Lakehouse Apps, set the DATABRICKS_HOST environment variable to the name of your Databricks account."
    assert defined('DATABRICKS_TOKEN') or (defined('DATABRICKS_CLIENT_ID') and defined('DATABRICKS_CLIENT_SECRET')), "To run outside of Lakehouse Apps, set environment variables for authentication, such as DATABRICKS_TOKEN or DATABRICKS_CLIENT_ID/DATABRICKS_CLIENT_SECRET."
    
    cfg = Config() # Pull environment variables for auth
    with sql.connect(server_hostname=os.getenv("DATABRICKS_HOST"),
                     http_path=f"""/sql/1.0/warehouses/{os.getenv("DATABRICKS_WAREHOUSE_ID")}""",
                     credentials_provider=lambda: cfg.authenticate) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
        
def insert_dataframe_to_table_old(df: pd.DataFrame, table_name: str):
    # Create table with appropriate schema
    columns = ", ".join([f"{col} STRING" for col, dtype in zip(df.columns, df.dtypes)])
    create_table_query = f"CREATE OR REPLACE TABLE {table_name} ({columns})"
    sqlQuery(create_table_query)
    
    # Insert data into the table
    for _, row in df.iterrows():
        values = ", ".join([f"'{str(val)}'" for val in row])
        insert_query = f"INSERT INTO {table_name} VALUES ({values})"
        sqlQuery(insert_query)

def insert_dataframe_to_table(df: pd.DataFrame, table_name: str):
    # Create table with appropriate schema
    columns = ", ".join([f"{col} {map_dtype(dtype)}" for col, dtype in zip(df.columns, df.dtypes)])
    create_table_query = f"CREATE OR REPLACE TABLE {table_name} ({columns})"
    sqlQuery(create_table_query)
    
    # Insert data into the table
    for _, row in df.iterrows():
        values = ", ".join([f"'{str(val)}'" for val in row])
        insert_query = f"INSERT INTO {table_name} VALUES ({values})"
        sqlQuery(insert_query)

def map_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INT"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    elif pd.api.types.is_string_dtype(dtype):
        return "STRING"
    elif pd.api.types.is_categorical_dtype(dtype):
        return "STRING"
    elif pd.api.types.is_timedelta64_dtype(dtype):
        return "INTERVAL"
    elif pd.api.types.is_complex_dtype(dtype):
        return "STRING"  # Complex numbers are not directly supported in SQL
    elif pd.api.types.is_decimal_dtype(dtype):
        return "DECIMAL"
    elif pd.api.types.is_date_dtype(dtype):
        return "DATE"
    elif pd.api.types.is_binary_dtype(dtype):
        return "BINARY"
    else:
        return "STRING"

def generate_fraud_record(schema_distribution):
    return eval(schema_distribution)

# Generate multiple synthetic records
def generate_fraud_data(num_records, schema_distribution):
    data = []
    for _ in range(num_records):
        data.append(generate_fraud_record(schema_distribution))
    return data

#remove special characters from dataframe
def remove_special_characters(df: pd.DataFrame) -> pd.DataFrame:
    # Define a function to remove special characters except for '.'
    def clean_string(s):
        return re.sub(r'[^A-Za-z0-9. ]+', '', s)
    
    # Apply the function to all string columns in the DataFrame
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: clean_string(x) if isinstance(x, str) else x)
    
    return df

# Streamlit app
st.title(":gray[DemoArigato - English to Databricks Demo]")
url = 'https://databricks.zoom.us/clips/share/bFgjBNr_c8KFRwCa7VTKQCse95Y5uLNlalgtG9TFeZPSwWGzXCnBs6OJv9FrnvSjbrkeDQ.pQrxYKWO3gnVMRoa'
st.subheader(":gray[Author: Praveen Gottam]", help='Video Tutorial at [link](%s). Demo beings at 1 minute mark' % url)

#    .stApp [data-testid="stToolbar"] {
#        display: none;
#    }

# Custom CSS to change background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9f7f4;
    }
    .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
    </style>
    """,
    unsafe_allow_html=True
)

# Input for the prompt template
prompt_template = st.text_input(":gray[Enter your business use case (ie: claims management for financial personas)]:", help='Your demo will be generated underneath the button after clicking the generate my demo button')

if st.button("Generate My Demo"):
    st.subheader(':gray[Please wait 3-4 minutes while we generate your demo and data below]')
    st.divider()
    try:
        # Generate use case
        st.header(":gray[Use Cases:]")
        use_case_prompt = PromptTemplate(
            input_variables=["business_question"],
            template="I need prompts to help me on {business_question} Please use domain specific knowledge that only employees in that industry would know. Just give 1 example. I want the example to be something that can be solved through data analysis of a dataset you will generate later. Please include industry specific terms. No need to create any sample data yet here. Do not ask me any follow up questions like to have you build a dataset."
        )
        use_case = run_chain(use_case_prompt, business_question=prompt_template)
        st.markdown(":gray["+str(use_case)+"]")
        st.divider()

         # Genereate Storyline
        st.header(":gray[Storyline]")
        storyline_prompt = PromptTemplate(
            input_variables=["use_case"],
            template="Create a story based off the context from {use_case} Make it professional as I will be using it for my actual professional work, Keep it short and brief with only the key information. No need to create any sample data yet here. Do not ask about it either yet. Do not ask me any follow ups."
        )
        storyline = run_chain(storyline_prompt, use_case=use_case)
        st.markdown(f'{storyline}')
        st.divider()

        # Generate sample data
        st.header(":gray[Sample Data:]")
        general_schema_prompt = PromptTemplate(
            input_variables=["use_case"],
            template="Based off {use_case} Build me a general schema related to it. Limit it to 5-7 columns All schema titles must have _ instead of spaces if it's two or more words. No other special characters allowed. No slashes. Have a mix between common fields that a worker would see with columns only workers would know"
        )
        general_schema = run_chain(general_schema_prompt, use_case=use_case)
        st.markdown(f':gray[{general_schema}]')
        st.divider()

        # Generate Genie Room Questions
        st.header(":gray[Genie Room Questions:]")
        st.subheader(":gray[Head over to Genie to create the room based off the synthetic data]")
        genie_data_prompt = PromptTemplate(
            input_variables=["general_schema"],
            template="I have a text-to-sql demo. What are some example questions that employees would ask? Give me 10 max. Only generate the questions themselves, no extra text. Consider that they may begin with exploratory / summary questions and then drill down to any outliers and then breakouts by another property. This is the data schema you have to work with: {general_schema}"
        )
        genie_questions = run_chain(genie_data_prompt, general_schema=general_schema)
        st.markdown(f':gray[{genie_questions}]')
        st.divider()

        # Genereate Lakeview Dashboard Prompts
        st.header(":gray[Lakeview Dashboard Prompts:]")
        st.subheader(":gray[Head over to Lakeview Dashboards to create the dashboard based off the synthetic data]")
        lakeview_prompt = PromptTemplate(
            input_variables=["genie_questions", "general_schema"],
            template="Based off the questions from {genie_questions}, what 4 key simple graphs should I make for regular users? YOU MUST provide it in a format that I can copy and paste into a english-to-visualization tool. Ex: (English-to-Visualization text: Bar chart of Risk Tolerance with count of Client ID on the y-axis and Risk Tolerance categories on the x-axis.) Do not include anything that needs a geolocation map. You can only create graphs based off this schema: {general_schema}"
        )
        lakeview_tips = run_chain(lakeview_prompt, genie_questions=genie_questions, general_schema=general_schema)
        st.markdown(f':gray[{lakeview_tips}]')
        st.divider()

        try:
            #generate a distribution
            schema_distribution_prompt = PromptTemplate(
                input_variables=["general_schema"],
                template="Based off {general_schema} what distributions do you expect of each of the columns? We can only use the faker library and numpy library to make our dataset. An example of what you should return is a dictionary with the column as the key and the corresponding expected distribution. I only want the JSON return. No other details, no other new lines or spacing.: \
                \
                {{'transaction_id': fake.uuid4(), 'customer_name': fake.name(), 'transaction_date': fake.date_this_year(), 'fraudulent': np.random.binomial(n=1, p=0.02), 'transaction_amount': round(np.random.lognormal(mean=5, sigma=1.5), 2)}}"
            )
            schema_distribution = run_chain(schema_distribution_prompt, general_schema=general_schema)

            #create fake data
            synthetic_data = generate_fraud_data(50, schema_distribution)
            synthetic_df = pd.DataFrame(synthetic_data)
            synthetic_df = remove_special_characters(synthetic_df)

            #get user info to register into Unity Catalog
            user = user_info.get('user_email').split('@')[0].replace(".", "_")

            #table for structured data
            database = 'demoarigato'
            table = '{user}_synthetic_data'
            #table_name = f"{catalog}.{database}.{table}"
            table_name = f"main.demoarigato.{user}_synthetic_data"

            # Write the DataFrame to a table in Unity Catalog
            insert_dataframe_to_table(synthetic_df, table_name)
            st.divider()
            st.markdown(f'Table successfully registered under Unity Catalog as : {table_name}')

            st.markdown(f':gray[Get running at : https://e2-demo-field-eng.cloud.databricks.com/explore/data/main/{database}/{user}_synthetic_data ]')
        
        except Exception as e:
            st.error(e)

    except json.JSONDecodeError:
        st.error("Invalid JSON format for additional arguments.")
