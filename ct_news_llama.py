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
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
# from gnews import GNews
import pandasai as pai
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
from clinical_trials_module import get_clinical_trials_data
from collections import Counter
from pandasai.llm import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.connectors import PandasConnector



normal_page = st.Page("ct_news_llama.py", title="Standard CT QnA", icon="☺️")
pro_page = st.Page("pages/Pro_version.py", title="Pro CT QnA", icon="😎")

pg = st.navigation([normal_page, pro_page])
# st.set_page_config()
st.title('CT Analysis')
st.set_page_config(page_title="CT Analysis", page_icon="💭",layout='wide', initial_sidebar_state='expanded')
pg.run()

# Loading config file
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Creating the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

#Setup mongoDB authentication
uri_mdb = "mongodb+srv://postlytllp:HGlyKh6SQfpqlejf@postlyt-test.l88dp2e.mongodb.net/?retryWrites=true&w=majority&appName=postlyt-test"
client = MongoClient(uri_mdb, server_api=ServerApi('1'))

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
            st.write(f'Welcome *{st.session_state["name"]}*')
            st.experimental_rerun()
        elif st.session_state.get("authentication_status") is False:
            st.error('Username/password is incorrect')
        elif st.session_state.get("authentication_status") is None:
            st.warning('Please Login to use GenAI for data analysis')

    with tab2:
        # Creating a new user registration widget
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                st.success('User registered successfully')
        except RegisterError as e:
            st.error(e)
    
    with tab3:
        # Creating a forgot password widget
        try:
            username_of_forgotten_password, email_of_forgotten_password, new_random_password = authenticator.forgot_password()
            if username_of_forgotten_password:
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


    # df=load_data(base_url,params)

    # # Convert the object to date and time
    # df["CompletionDate"]=df["CompletionDate"].apply(
    #     lambda x :pd.to_datetime(
    #         x[0]) if isinstance(
    #         x, list) and len(
    #         x) > 0 else pd.NaT )

    # df["StartDate"]=df["StartDate"].apply(
    #     lambda x :pd.to_datetime(
    #         x[0]) if isinstance(
    #         x, list) and len(
    #         x) > 0 else pd.NaT )

    # # Counting the number of trials going on
    # df["Nos_location"]=df["LocationCountry"].apply(lambda x: len(x) if isinstance(x,list) else 0)

    # return df

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
        
    df = get_clinical_trials_data(st.session_state.text)
    drop_columns= [
                   "organizationType","officialTitle","statusVerifiedDate","hasExpandedAccess", "studyFirstSubmitDate",
                   "studyFirstPostDate", "lastUpdatePostDate", "lastUpdatePostDateType", "HasResults", "responsibleParty",
                   "collaboratorsType", "briefSummary", "detailedDescription", "allocation","interventionModel",
                   "primaryPurpose","masking","whoMasked","enrollmentCount","enrollmentType","arms",
                   "interventioOthers","interventionDescription","primaryOutcomes","secondaryOutcomes",
                   "eligibilityCriteria","healthyVolunteers","eligibilityGender","eligibilityMinimumAge",
                   "eligibilityMaximumAge","eligibilityStandardAges"
                  ]
    df=df.drop(columns=drop_columns)
  
     #Heading for sidebar
    st.sidebar.header('CT Dashboard `v0.3`')

    #selecting the study type
    df['StudyType_str']=df.loc[:,'studyType'].apply( lambda x: 'N/A' if len(x)==0 else ' '.join(map(str,x)))
    options_st = df['StudyType_str'].unique().tolist()
    options_st.insert(0, "All")
    selected_options_str = st.sidebar.selectbox('What kind of study you want?',options= options_st)

    if selected_options_str == 'All':
        filtered_df = df
    else:
         #Convert selected_options_str back to a list
        selected_options = selected_options_str.split()
        filtered_df= df[df.StudyType_str.isin(selected_options)]
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
      
        # # Rename columns
        # top_5_interventions_df = top_5_interventions_df.rename(columns={'index': 'Interventions', 0: 'Count'})

       

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
    df_1 = df_genai.copy()

    # for col in df_1.columns:
    #     if pd.api.types.is_integer_dtype(df_1[col]):
    #         pass
    #     elif (pd.api.types.is_datetime64_any_dtype(df_1[col]) or pd.api.types.is_bool_dtype(df_1[col])):
    #         pass
    #     elif pd.api.types.is_bool_dtype(df_1[col]):
    #         pass  
    #     else:
    #         df_1[col] = df_genai.loc[:,col].apply(lambda x: 'N/A' if len(x) == 0 else ' '.join(map(str, x)))

    # Iterate through columns of df_1
    for col in df_1.columns:
        if pd.api.types.is_integer_dtype(df_1[col]):
            pass
        elif pd.api.types.is_datetime64_any_dtype(df_1[col]):
            pass
        elif pd.api.types.is_bool_dtype(df_1[col]):
            pass
        else:
            df_1[col] = df_genai[col].apply(lambda x: 'N/A' if not x else ' '.join(map(str, x)) if isinstance(x, list) else x)  
    
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

    if df_1 is not None:
        # #row C
        # t1,t2=st.columns((7,3))

        # with t1:
            #
            # def _submit_feedback(user_response, emoji=None):
            #     st.info(f"Feedback submitted: {user_response}", icon=emoji)
            #     return user_response.update({"some metadata": 123})


        pai.clear_cache()
        # llm = ChatGroq(model_name='llama3-70b-8192', api_key="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")
        # llm = OpenAI(
        #               organization:'org-7VsS9klsOkUjb3FkcvQzxn2l',
        #               project:'proj_9RDJQE1vAIuNz2fIPeF5cho8',
        #               model: "gpt-3.5-turbo-0125",
        #               OPENAI_API_KEY: 'sk-proj-XKudWYOe0DrzebixiEhST3BlbkFJTrpK0LkXbBkIOzN2Zq1h'
        #             )

        field_descriptions = {
                              "nctId": "The unique identifier for each clinical trial registered on ClinicalTrials.gov.",
                              "organization": "The name of the organization conducting the clinical trial.",
                              "briefTitle": "A short title for the clinical trial, intended for easy reference.",
                              "overallStatus": "The current overall status of the clinical trial ('COMPLETED', 'UNKNOWN',/n" 
                              "'ACTIVE_NOT_RECRUITING', 'RECRUITING', 'WITHDRAWN', 'TERMINATED', 'ENROLLING_BY_INVITATION', 'NOT_YET_RECRUITING', 'APPROVED_FOR_MARKETING',).",
                              "startDate": "The date when the clinical trial began.",
                              "completionDate": "The date when the clinical trial was completed.",
                              "completionDateType": "The type of completion date, specifying whether it refers to the ACTUAL or ESTIMATED.",
                              "leadSponsor": "The primary sponsor responsible for the initiation, management, and financing of the clinical trial.",
                              "leadSponsorType": "The type of the lead sponsor, such as academic, industry, or government.",
                              "collaborators": "Other organizations or individuals collaborating on the clinical trial.",
                              "conditions": "The medical conditions or diseases being studied in the clinical trial.",
                              "studyType": "The type of study (e.g., interventional, observational).",
                              "phases": "The phase of the clinical trial (e.g.,  'PHASE2', 'PHASE2, PHASE3', 'PHASE3', 'PHASE1', 'PHASE4','PHASE1, PHASE2', 'EARLY_PHASE1').",
                              "interventionDrug": "The drugs or medications being tested or used as interventions in the clinical trial.",
                              "LocationName": "The names of the locations where the clinical trial is being conducted.",
                              "city": "The city where the clinical trial locations are situated.",
                              "state": "The state where the clinical trial locations are situated.",
                              "country": "The country where the clinical trial locations are situated.",
                              "interventionBiological": "Biological interventions (e.g., vaccines, blood products) used in the clinical trial."
                            }

      
        llm = ChatGroq(model_name='llama3-70b-8192', api_key="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")
        # config_llm=({'llm': llm,'llm_options':{'model':'gpt-4o'},'response_parse': StreamlitResponse,'verbose':True,'max_retries':5})
        connector = PandasConnector({"original_df": df_1}, field_descriptions=field_descriptions)
        config_llm=({'llm': llm, 'llm_options':{'model':'gpt-4o'},'verbose': True,'response_parse': StreamlitResponse,"original_df": df_1})
        df_smart = SmartDataframe(connector, config=config_llm)
    
        
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
                
    
        if prompt := st.chat_input(placeholder="What is this data about?",disabled= chat_disable):
    
            # Create a new feedback entry
            # st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.prompt = prompt
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
    
            with st.chat_message("assistant"):
                response = df_smart.chat(prompt)
                st.write(response)
                
    
                # to show chart
                st.set_option('deprecation.showPyplotGlobalUse', False)
    
                if plt.get_fignums():
                    st.pyplot()
    
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.answer = response
                    
    
        # st.rerun()
    
        # Clear conversation history button
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
            
            
