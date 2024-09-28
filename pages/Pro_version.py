#%%writefile app.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:
# pip install plost

import streamlit as st
import requests
import pandas as pd
import plost
import os
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import matplotlib.pyplot as plt
from st_btn_group import st_btn_group
from streamlit_feedback import streamlit_feedback
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                          ForgotError,
                                                          LoginError,
                                                          RegisterError,
                                                          ResetError,
                                                          UpdateError)
from streamlit_authenticator.utilities.hasher import Hasher
from streamlit_authenticator.utilities.helpers import Helpers
from streamlit_authenticator.utilities.validator import Validator
from clinical_trials_module import get_clinical_trials_data
from collections import Counter
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
import numpy as np

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.title('CT Analysis')

# Loading config file
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

# # Creating the authenticator object
# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['pre-authorized']
# )

#Setup mongoDB authentication
uri_mdb = "mongodb+srv://postlytllp:HGlyKh6SQfpqlejf@postlyt-test.l88dp2e.mongodb.net/?retryWrites=true&w=majority&appName=postlyt-test"
client = MongoClient(uri_mdb, server_api=ServerApi('1'))
db = client['test1']
collection_user = db['user_details']

def get_credentials_from_mongo():
    
    users = collection_user.find({})
    credentials = {"usernames": {}}

    for user in users:
        credentials["usernames"][user["User_ID"]] = {
            "email": user["email"],
            "failed_login_attempts": user.get("failed_login_attempts", 0),
            "logged_in": user.get("logged_in", False),
            "name": user["name"],
            "password": user["password"]
        }
    
    return credentials

# Save user data to MongoDB (for registration, reset password, etc.)
def update_user_in_mongo(user_id, update_data):
    collection_user.update_one({"User_ID": user_id}, {"$set": update_data})

# Loading config file
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Fetching the credentials from MongoDB
credentials_from_db = get_credentials_from_mongo()

#login function
# Creating the authenticator object
authenticator = stauth.Authenticate(
    credentials_from_db,
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)


#login function
@st.experimental_dialog("Login")
def show_authentication_ui():
    tab1, tab2, tab3, tab4 = st.tabs(["Login", "Register", "Forgot Password", "Update Details"])
    
    with tab1:
        # Creating a login widget
        try:
            authenticator.login()
        except LoginError as e:
            st.error(e)
        
        if st.session_state.get("authentication_status"):
            authenticator.logout()
            # st.write(f'Welcome *{st.session_state["username"]}*')
            st.rerun()
        elif st.session_state.get("authentication_status") is False:
            st.error('Username/password is incorrect')
        elif st.session_state.get("authentication_status") is None:
            st.warning('Please Login to use GenAI for data analysis')

    with tab2:
        # Creating a new user registration widget
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user, new_password  = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                # Insert new user into MongoDB
                hashed_password = Hasher._hash(new_password)
                # hashed_password = new_password
                collection_user.insert_one({
                    "User_ID": username_of_registered_user,
                    "email": email_of_registered_user,
                    "name": name_of_registered_user,
                    "password": hashed_password,
                    "failed_login_attempts": 0,
                    "logged_in": False,
                    "pro_user" : False
                })
                st.success('User registered successfully')
                # st.rerun()
        except RegisterError as e:
            st.error(e)
    
    with tab3:
        # Creating a forgot password widget
        try:
            username_of_forgotten_password, email_of_forgotten_password, new_random_password = authenticator.forgot_password()
            if username_of_forgotten_password:
                # Update MongoDB with the new password
                # new_password = Hasher._hash(st.session_state['new_password'])
                new_password = Hasher._hash(st.session_state['new_password'])
                update_user_in_mongo(st.session_state["username"], {"password": new_password})
                st.success('New password sent securely')
                # Ensure the random password is transferred to the user securely
            else:
                st.error('Username not found')
        except ForgotError as e:
            st.error(e)

    with tab4:
        # Creating an update user details widget
        if st.session_state.get("authentication_status"):
            try:
                if authenticator.update_user_details(st.session_state["username"]):
                    # Update the MongoDB with new details
                    update_user_in_mongo(st.session_state["username"], {
                        "name": st.session_state["new_name"],
                        "email": st.session_state["new_email"]
                    })
                    st.success('Entries updated successfully')
            except UpdateError as e:
                st.error(e)
        else:
            st.warning('Please log in to update your details')
    
    # Saving config file
    with open('config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False)

def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


def extract_info(intervention_str):
    parts = intervention_str.split(": ")
    intervention_type = parts[0]  # Get the first part (e.g., "Biological")
    intervention_name = ": ".join(parts[1:])  # Join the rest (e.g., "Experimental : Secukinumab low dose")
    return [intervention_type, intervention_name]

if  'text' not in st.session_state:
    st.session_state.CONNECTED =  False
    st.session_state.text = ''


def _connect_form_cb(connect_status):
    st.session_state.CONNECTED = connect_status


def display_db_connection_menu():
    with st.form(key="connect_form"):
        st.text_input('Enter the condition', help='Click on search, pressing enter will not work', value=st.session_state.text, key='text')
        submit_button = st.form_submit_button(label='Search', on_click=_connect_form_cb, args=(True,))
        if submit_button:
            if st.session_state.text=='':
                st.write("Please enter a condition")
                st.stop()



display_db_connection_menu()

global chat_disable

if st.session_state.CONNECTED:

    st.write('You are Searching for:',  st.session_state.text)
    if st.session_state["authentication_status"]:
        st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
        chat_disable=False
        with st.sidebar:
            authenticator.logout()
    else:
        st.sidebar.button("Login",type="primary", on_click=show_authentication_ui())
        st.info('Login to use GenAI to get answers', icon="👤")
        chat_disable=True        
        
    df_1 = get_clinical_trials_data(st.session_state.text)
    # drop_columns= [
    #                "organizationType","officialTitle","statusVerifiedDate","hasExpandedAccess", "studyFirstSubmitDate",
    #                "studyFirstPostDate", "lastUpdatePostDate", "lastUpdatePostDateType", "HasResults", "responsibleParty",
    #                "collaboratorsType", "briefSummary", "detailedDescription", "allocation","interventionModel",
    #                "primaryPurpose","masking","whoMasked","enrollmentCount","enrollmentType","arms",
    #                "interventioOthers","interventionDescription","primaryOutcomes","secondaryOutcomes",
    #                "eligibilityCriteria","healthyVolunteers","eligibilityGender","eligibilityMinimumAge",
    #                "eligibilityMaximumAge","eligibilityStandardAges"
    #               ]
    # df_1=df_1.drop(columns=drop_columns)
  
     #Heading for sidebar
    st.sidebar.header('CT Dashboard `v0.3`')

    #selecting the study type
    df_1['StudyType_str']=df_1.loc[:,'studyType'].apply( lambda x: 'N/A' if len(x)==0 else ' '.join(map(str,x)))
    options_st = df_1['StudyType_str'].unique().tolist()
    options_st.insert(0, "All")
    selected_options_str = st.sidebar.selectbox('What kind of study you want?',options= options_st)

    if selected_options_str == 'All':
        filtered_df = df_1
    else:
         #Convert selected_options_str back to a list
        selected_options = selected_options_str.split()
        filtered_df= df_1[df_1.StudyType_str.isin(selected_options)]
        pass
        if filtered_df.empty:
            st.write("No studies found for the selected options.")
            st.stop()

    # Slider for selecting year and month
    st.sidebar.subheader('Start Year of CT')
    years = filtered_df['startDate'].dt.year.unique()
    selected_year_range = st.sidebar.slider('Select Year Range', min_value=int(min(years)), max_value=int(max(years)), value=(int(min(years)), int(max(years))), key='slider_year')


    # Filter the DataFrame based on the selected dates
    filtered_df = filtered_df[(filtered_df['startDate'].dt.year >= selected_year_range[0]) & (filtered_df['startDate'].dt.year <= selected_year_range[1])]
    # filtered_df['startDate'] = filtered_df['startDate'].dt.strftime('%Y-%m')
    filtered_df['phases']=filtered_df['phases'].fillna('N/A')



    #Data for pie chart
    filtered_df.loc[:,'Phase_str'] = filtered_df.loc[:,'phases'].apply(lambda x: 'N/A' if len(x) == 0 else ' '.join(map(str, x)))
    filtered_df_pie=filtered_df.groupby("Phase_str")['nctId'].count().rename('count_phase').reset_index()


    #Select the Phase for pie chart
    options = filtered_df_pie['Phase_str'].unique().tolist()
    selected_options = st.sidebar.multiselect('Which app do you want?',options)

    if "subscriber" not in st.session_state:
          st.session_state.subscriber = False
    if st.session_state.subscriber == False:
      st.sidebar.link_button("Click here for subscription☕","https://www.buymeacoffee.com/Shubh789",type="primary")
      
    st.sidebar.markdown('''
    ---
      Created with ❤️ by [Shubhanshu](https://www.linkedin.com/in/shubh789/).
    ''')

    #data for side bars
    filtered_df["StartYear"]=filtered_df['startDate'].dt.year
    if len(selected_options) == 0:
        filtered_df_lc=filtered_df.groupby('StartYear')['nctId'].count().rename('Nos_CT').reset_index()

    else:
        filtered_df_lc_pie = filtered_df[filtered_df.Phase_str.isin(selected_options)]
        filtered_df_lc = filtered_df_lc_pie.groupby('StartYear')['nctId'].count().rename('Nos_CT').reset_index()



    # Row A
    st.markdown('### Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric("Nos. of studies", filtered_df_lc.Nos_CT.sum())

    #Nos. of recruiting studies
    recruiting_count = (
    filtered_df[filtered_df['overallStatus'] == 'RECRUITING']['nctId'].count()
    if len(selected_options) == 0
    else filtered_df_lc_pie[filtered_df_lc_pie['overallStatus'] == 'RECRUITING']['nctId'].count()
    )

    col2.metric("Nos. Recruiting CT", recruiting_count)

    #Nos. of completed studies
    completion_count = (
    filtered_df[filtered_df['completionDateType'] == 'ACTUAL']['nctId'].count()  
    if len(selected_options) == 0
    else filtered_df_lc_pie[filtered_df_lc_pie['completionDateType'] == 'ACTUAL']['nctId'].count()
    )
    col3.metric("Trials completed", completion_count)

    #row B
    c1,c2,c3,c4=st.columns((1,4,4,1))

    with c1:
        st.container()

    with c2:
        st.markdown('##### Top 5 industry sponsors')
        sponsors = filtered_df[filtered_df['leadSponsorType'] == 'INDUSTRY']['leadSponsor'].to_frame()
        # Group by 'leadSponsor' and count the occurrences
        sponsor_counts = sponsors['leadSponsor'].value_counts().reset_index()
        sponsor_counts.columns = ['Sponsors', 'Count']
        # Get the top 5 sponsors by count
        top5_sponsors = sponsor_counts.head(5)
        st.table(top5_sponsors)

    if len(selected_options) == 0:
        filtered_pie = filtered_df_pie  # No filtering required, keep all data
    else:
        filtered_pie = filtered_df_pie[filtered_df_pie['Phase_str'].isin(selected_options)]


    with c3:
        st.markdown('##### Top 5 Interventions')
        # # Explode the list column to separate rows
        # df_exploded = filtered_df.explode('ArmGroupInterventionName')
        
        # # Split the intervention name and type
        # df_exploded.dropna(subset=['ArmGroupInterventionName'], inplace=True)
        
        # # Apply the function to extract information
        # df_exploded[['InterventionType', 'InterventionName']] = df_exploded['ArmGroupInterventionName'].apply(extract_info).tolist()
                
        # # Count intervention names
        # intervention_counts = df_exploded['InterventionName'].value_counts()

 
        # # Get the top 5 intervention names
        # top_5_interventions = intervention_counts.nlargest(5)
       
        # # Convert Series to DataFrame
        # top_5_interventions_df = top_5_interventions.reset_index()
        
        # Combine the two columns and split by comma
        combined_entities = (
            filtered_df['interventionDrug'].fillna('') + ',' + 
            filtered_df['interventionBiological'].fillna('')
        )
        all_entities = [entity.strip() for row in combined_entities for entity in row.split(',') if entity.strip()]
        
        # Count occurrences and get top 5
        entity_counts = Counter(all_entities)
        top_5_entities = entity_counts.most_common(5)
        
        # Create a new dataframe
        top_5_interventions_df = pd.DataFrame(top_5_entities, columns=['Drug/Biological', 'Count'])
      

       

        # Print the top 5 interventions
        st.table(top_5_interventions_df)

    dataExploration = st.container()

    with dataExploration:
#       st.title('Clinical trials data')
      st.subheader('Sample data')
#       st.header('Dataset: Clinical trials of', st.session_state.text)
      st.markdown('I found this dataset at... https://clinicaltrials.gov')
      st.markdown('**It is a sample of 100 rows from the dataset**')
#       st.text('Below is the sample DataFrame')
      st.dataframe(filtered_df.head(100))


    # Assuming you have your service account JSON key file in your project directory
    key_path = "test-gemini-420318-e15da8ebca2c.json"

    # Check if the environment variable is already set (prevents accidental override)
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    df_genai = filtered_df.drop(columns=['StudyType_str', 'Phase_str'])
    df_genai.rename(columns = {'nctId':'Trials ID'}, inplace = True)
    
    # Apply to all columns
    df = df_genai.copy()

    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            pass
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            pass
        elif pd.api.types.is_bool_dtype(df[col]):
            pass
        else:
            df[col] = df_genai[col].apply(lambda x: 'N/A' if not x else ' '.join(map(str, x)) if isinstance(x, list) else x)  
    
    # Function to handle feedback submission
    def _submit_feedback(feedback_key,feedback):
        db = client['test1']
        collection = db['postlyt-test']

        # Retrieve prompt and answer associated with the feedback_key from session state
        prompt = st.session_state.feedbacks.get(feedback_key, {}).get('prompt', '')
        answer = st.session_state.feedbacks.get(feedback_key, {}).get('answer', '')
        
        # Update existing feedback with score and comment
        collection.update_one(
            {"feedback_key": feedback_key},
            {
                "$set": {
                    "score": feedback['score'],
                    "Comment": feedback['text'],
                    "User": st.session_state["name"]
                }
            }
        )

        st.info(f"Feedback submitted: {feedback}, {prompt}, A: {answer}", icon="👍")

    if "clear" not in st.session_state:
        st.session_state.clear = False
            
    if "prompt" not in st.session_state:
        st.session_state.prompt=[]
        
    if "answer" not in st.session_state:
        st.session_state.answer=[]    

    if "messages" not in st.session_state or st.session_state["clear"]:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state.feedbacks = {}  # Ensure feedbacks are reset when messages are cleared
        st.session_state.clear = False  # Reset the clear flag immediately after clearing messages



    # Initialize feedback state OUTSIDE the loop
    if "feedbacks" not in st.session_state:
        st.session_state.feedbacks = {}

    if df is not None:
        instruction_str = (
            "1. Convert the query to executable Python code using Pandas.\n"
            "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
            "3. The code should represent a solution to the query.\n"
            "4. PRINT ONLY THE EXPRESSION.\n"
            "5. Do not quote the expression.\n"
            "6.The column headings and its description are given\n"
              "nctId: The unique identifier for each clinical trial registered on ClinicalTrials.gov.\n"
              "organization:  The name of the organization conducting the clinical trial.\n"
              "organizationType:  The type of organization, such as 'OTHER', 'INDUSTRY', 'NIH', 'OTHER_GOV', 'INDIV', 'FED', 'NETWORK', 'UNKNOWN'.\n"
              "briefTitle:  A short title for the clinical trial, intended for easy reference.\n"
              "officialTitle:  The full official title of the clinical trial.\n"
              "statusVerifiedDate:  The date when the status of the clinical trial was last verified.\n"
              "overallStatus:  The current overall status of the clinical trial like 'COMPLETED', 'UNKNOWN', 'ACTIVE_NOT_RECRUITING', 'RECRUITING', 'WITHDRAWN', 'TERMINATED', 'ENROLLING_BY_INVITATION', 'NOT_YET_RECRUITING', 'APPROVED_FOR_MARKETING', 'SUSPENDED','AVAILABLE'.\n"
              "hasExpandedAccess:  It has boolean values and it indicates whether the clinical trial includes expanded access to the investigational drug or device outside of the clinical trial.\n"
              "startDate:  The date when the clinical trial began.\n"
              "completionDate:  The date when the clinical trial was completed.\n"
              "completionDateType:  The type of completion date, specifying whether it refers to the ACTUAL or ESTIMATED.\n"
              "studyFirstSubmitDate:  The date when the clinical trial information was first submitted to ClinicalTrials.gov.\n"
              "studyFirstPostDate:  The date when the clinical trial information was first posted on ClinicalTrials.gov.\n"
              "lastUpdatePostDate:  The date when the clinical trial information was last updated on ClinicalTrials.gov.\n"
              "lastUpdatePostDateType:  The type of last update post date, specifying whether it refers to the actual or anticipated date.\n"
              "HasResults:  It contains boolean values and indicates whether the results of the clinical trial have been posted on ClinicalTrials.gov.\n"
              "responsibleParty:  The individual or organization responsible for the overall conduct of the clinical trial.\n"
              "leadSponsor:  The primary sponsor responsible for the initiation, management, and financing of the clinical trial.\n"
              "leadSponsorType:  The type of the lead sponsor, such as academic, industry, or government.\n"
              "collaborators:  Other organizations or individuals collaborating on the clinical trial.\n"
              "collaboratorsType:  The types of collaborators involved in the clinical trial.\n"
              "briefSummary:  A brief summary of the clinical trial, providing an overview of the study's purpose and key details.\n"
              "detailedDescription:  A detailed description of the clinical trial, including comprehensive information about the study design, methodology, and objectives.\n"
              "conditions:  The medical conditions or diseases being studied in the clinical trial.\n"
              "studyType:  The type of study (e.g., 'INTERVENTIONAL', 'OBSERVATIONAL', 'EXPANDED_ACCESS').\n"
              "phases:  The phase of the clinical trial (e.g., 'NA', 'PHASE2', 'PHASE2, PHASE3', 'PHASE3', 'PHASE1', 'PHASE4','PHASE1, PHASE2', 'EARLY_PHASE1').\n"
              "allocation:  The method of assigning participants to different arms of the clinical trial (e.g., 'RANDOMIZED','NON_RANDOMIZED').\n"
              "interventionModel:  The model of intervention used in the clinical trial (e.g., 'SINGLE_GROUP', 'PARALLEL', 'CROSSOVER', 'SEQUENTIAL', 'FACTORIAL').\n"
              "primaryPurpose:  The primary purpose of the clinical trial like 'PREVENTION', 'TREATMENT', 'SUPPORTIVE_CARE','BASIC_SCIENCE', 'DIAGNOSTIC', 'OTHER', 'ECT', 'SCREENING','HEALTH_SERVICES_RESEARCH', 'DEVICE_FEASIBILITY').\n"
              "masking:  The method used to prevent bias by concealing the allocation of participants (e.g., 'QUADRUPLE', 'NONE', 'DOUBLE', 'TRIPLE', 'SINGLE').\n"
              "whoMasked:  Specifies who is masked in the clinical trial etc. PARTICIPANT, INVESTIGATOR etc).\n"
              "enrollmentCount:  The number of participants enrolled in the clinical trial.\n"
              "enrollmentType:  The type of enrollment, specifying whether the number is ACTUAL or ESTIMATED.\n"
              "arms:  The number of arms or groups in the clinical trial.\n"
              "interventionDrug:  The drugs or medications being tested or used as interventions in the clinical trial.\n"
              "interventionDescription:  Descriptions of the interventions used in the clinical trial.\n"
              "interventionOthers:  Other types of interventions used in the clinical trial (e.g., devices, procedures).\n"
              "primaryOutcomes:  The primary outcome measures being assessed in the clinical trial.\n"
              "secondaryOutcomes:  The secondary outcome measures being assessed in the clinical trial.\n"
              "eligibilityCriteria:  The criteria that determine whether individuals can participate in the clinical trial.\n"
              "healthyVolunteers:  Indicates whether healthy volunteers are accepted in the clinical trial.\n"
              "eligibilityGender:  The gender eligibility criteria for participants in the clinical trial.\n"
              "eligibilityMinimumAge:  The minimum age of participants eligible for the clinical trial.\n"
              "eligibilityMaximumAge:  The maximum age of participants eligible for the clinical trial.\n"
              "eligibilityStandardAges:  Standard age groups eligible for the clinical trial.\n"
              "LocationName:  The names of the locations where the clinical trial is being conducted.\n"
              "city:  The city where the clinical trial locations are situated.\n"
              "state:  The state where the clinical trial locations are situated.\n"
              "country:  The country where the clinical trial locations are situated.\n"
              "interventionBiological:  Biological interventions (e.g., vaccines, blood products) used in the clinical trial."
           
        )
        
        pandas_prompt_str = (
            "You are working with a pandas dataframe in Python.\n"
            "The name of the dataframe is `df`.\n"
            "This is the result of `print(df.head())`:\n"
            "{df_str}\n\n"
            "Follow these instructions:\n"
            "{instruction_str}\n"
            "Query: {query_str}\n\n"
            "Expression:"
        )
        response_synthesis_prompt_str = (
            "Given an input question, synthesize an answer for audience expert in clinical trial data from the query results.\n"
            "Query: {query_str}\n"
            "Do not add any extra generated text and include NCTIDs in answer if required\n"
            "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
            "Pandas Output: {pandas_output}\n\n"
            "Response: "
        )
        
        pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
            instruction_str=instruction_str, df_str=df.head(5)
        )
        pandas_output_parser = PandasInstructionParser(df)
        response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
        llm = OpenAI(api_key="sk-proj-XKudWYOe0DrzebixiEhST3BlbkFJTrpK0LkXbBkIOzN2Zq1h",model="gpt-4o")
        # llm_code = OpenAI(api_key="sk-proj-XKudWYOe0DrzebixiEhST3BlbkFJTrpK0LkXbBkIOzN2Zq1h",model="o1-preview")

        qp = QP(
            modules={
                "input": InputComponent(),
                "pandas_prompt": pandas_prompt,
                "llm1": llm,
                "pandas_output_parser": pandas_output_parser,
                "response_synthesis_prompt": response_synthesis_prompt,
                "llm2": llm,
            },
            verbose=True,
        )
        qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
        qp.add_links(
            [
                Link("input", "response_synthesis_prompt", dest_key="query_str"),
                Link(
                    "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
                ),
                Link(
                    "pandas_output_parser",
                    "response_synthesis_prompt",
                    dest_key="pandas_output",
                ),
            ]
        )
        # add link from response synthesis prompt to llm2
        qp.add_link("response_synthesis_prompt", "llm2")
      
        # llm = ChatGroq(model_name='llama3-70b-8192', api_key="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")
        # # config_llm=({'llm': llm,'llm_options':{'model':'gpt-4o'},'response_parse': StreamlitResponse,'verbose':True,'max_retries':5})
        # connector = PandasConnector({"original_df": df}, field_descriptions=field_descriptions)
        # config_llm=({'llm': llm, 'llm_options':{'model':'gpt-4o'},'verbose': True,'response_parse': StreamlitResponse,"original_df": df})
        # df_smart = SmartDataframe(connector, config=config_llm)
    
        
        feedback_counter = len(st.session_state["feedbacks"])  # Start from where we left off
    
        for n, msg in enumerate(st.session_state.messages):
            st.chat_message(msg["role"]).write(msg["content"])
    
            # Feedback handling moved INSIDE the loop
            if msg["role"] == "assistant" and n > 0: 
                feedback_key = f"feedback_{n}_{st.session_state.messages[n - 1]['content']}"
                # with st.container():  # Use a container to isolate feedback widgets
                   
                # Check if feedback is already submitted
                if st.session_state.get(f"feedback_submitted_{feedback_key}", False):
                    # If submitted, skip adding feedback widget
                    continue
                
                db = client['test1']
                collection = db['postlyt-test']
                
                # Store prompt and answer directly in session state under feedback_key (only once)
                st.session_state.feedbacks[feedback_key] = {
                    'prompt': st.session_state.messages[n - 1]['content'],
                    'answer': msg['content']
                }
                
                # *** Check if the document already exists before inserting ***
                if not collection.find_one({"feedback_key": feedback_key}):
                # Insert the prompt, answer, and feedback_key into MongoDB (only if it doesn't exist)
                    try:
                        collection.insert_one({
                            "feedback_key": feedback_key, 
                            "condition": st.session_state.get('text', ''),
                            # Use the potentially converted answer from session state
                            "prompt": st.session_state.feedbacks[feedback_key]['prompt'],  
                            "answer": st.session_state.feedbacks[feedback_key]['answer'],
                            "User": st.session_state["name"],
                            "score": None,
                            "Comment": None 
                        })
                    except pymongo.errors.InvalidDocument as e:
                        # st.error(f"Error storing answer in database: {e}")
                        st.session_state.feedbacks[feedback_key]['answer'] = "Error encoding answer"  # Store error message
                        collection.insert_one({  # Insert error document
                            "feedback_key": feedback_key, 
                            "condition": st.session_state.get('text', ''),
                            "prompt": st.session_state.feedbacks[feedback_key]['prompt'], 
                            "answer": f"Error: {e}",
                            "User": st.session_state["name"],
                            "score": None,
                            "Comment": None 
                        })
    
    
                streamlit_feedback(
                    feedback_type="thumbs",
                    optional_text_label="Please provide extra information",
                    on_submit=lambda feedback: _submit_feedback(feedback_key, feedback),
                    key=feedback_key,
                )
        # Query to get all users where pro_user is True and return only the userid field
        pro_user_ids = collection_user.find({"pro_user": True}, {"_id": 0, "User_ID": 1})
        
        # Convert the cursor into a list of user IDs
        pro_user = [user['User_ID'] for user in pro_user_ids]

        # pro_user=['Ashok Kumar Chenda','Shubh07','shubh']
      
        if st.session_state["username"] in pro_user:
          is_pro_user=True
          chat_msg = 'Ask questions regarding your query(Pro version)'
        else:
          is_pro_user=False
          chat_msg = 'Subscribe to PRO Version for advance quering'
      
        if prompt := st.chat_input(placeholder=chat_msg, disabled = not(is_pro_user)):
    
            # Create a new feedback entry
            # st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.prompt = prompt
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
    
            with st.chat_message("assistant"):
                response_q = qp.run(query_str=str(prompt))
                # response = df_smart.chat(prompt)
                response = str(response_q.message.content)
                st.write(response)
                
    
                # to show chart
                # st.set_option('deprecation.showPyplotGlobalUse', False)
    
                if plt.get_fignums():
                    st.pyplot()
    
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.answer = response
                    
    
        # st.rerun()
    
        # Clear conversation history button
        if st.session_state.subscriber==True:  
          if st.sidebar.button("Clear conversation history", type="primary"):
              # Clear messages and feedbacks
              st.session_state.clear = True
              
              for key in st.session_state.feedbacks:
                  st.session_state[f"feedback_submitted_{key}"] = False
              st.rerun()

        # with t2:


    

    
    # google_news = GNews()
    # @st.cache_data(show_spinner=False)  # Cache the results based on 'text'
    # def fetch_news(search_term):
    #     return google_news.get_news(search_term)
    
    # if 'text' in st.session_state:
    #     search_term = st.session_state.text
    
    #     # Fetch news (cached unless 'text' changes)
    #     ct_news = fetch_news(f"Lastest clinical trials for {search_term}")
    
    #     if ct_news:
    #         st.subheader(f"News for '{search_term}':")
    
    #         for article in ct_news[:10]:
    #             st.markdown(f"**{article['title']}**\n Published on: {article['published date']}  \n[Read more]({article['url']})")
    #     else:
    #         st.write(f"No news found for '{search_term}'.")
    # else:
    #     st.write("Please enter a search term.")

# elif authentication_status is False:
#     st.error("Authentication failed")
# elif authentication_status is None:
#     st.warning("Please log in")
            
            

