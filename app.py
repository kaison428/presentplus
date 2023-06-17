import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from model import *

# Set API Key
import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

st.set_page_config(page_title="PresentPlus - An LLM-powered Presentation Mentor", page_icon=":star:")

# Sidebar contents
with st.sidebar:
    st.title(':eye: PresentPlus')
    st.markdown('''
    ## About
    This app is an LLM-powered presentation analyzer built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/en/latest/)
    
    ''')

    add_vertical_space(5)
    st.write('Made by Kaison from HYPE AI :book:')

# Layout of input/response containers
input_container = st.container()
context_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# Flag
isPDF = False

# User input
## Function for taking user provided PDF as input
def get_file(key):
    uploaded_file = st.file_uploader(f"Upload your {key} file", type='pdf', key=key)
    if uploaded_file is None:
        st.session_state["upload_state"] = False
    else:
        st.session_state["upload_state"] = True
        
    return uploaded_file

## Applying the user input box
with input_container:
    user_input = get_file('presentation')

def get_text(prompt):
    input_text = st.text_area(prompt, "", key="input")
    return input_text

## Applying the user context box
with context_container:
    context_text = get_text('Input the problem statement: ')

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(fileobj, context_text):
    text = get_text_from_pdf(fileobj)
    summary = custom_prompt_summary(text, context_text, chain_type='map_reduce')
    recommendation = get_recommendation(text, context_text, chain_type='map_reduce')
    return summary, recommendation

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input and context_text:
        summary, recommendation = generate_response(user_input, context_text)
        output = f'''
            Summary:
                {summary}
            
            Recommendations:
                {recommendation}
        '''
        
        message(output)
