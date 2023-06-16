import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from model import *

# Set API Key
import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

################################################################
PROMPT_TEMPLATE = '''
    You are a professional in marketing, specializing in market research and technology development. \
    You are a judge of an undergraduate marketing presentation competition. \
    
    Write a concise summary of the following:\
    
    {text}

   The summary should focus on the technical content and technology aspect of presentation.\

'''

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
# context_container = st.container()
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

# ## Applying the user context box
# with context_container:
#     user_context = get_file('context')

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(fileobj):
    text = get_text_from_pdf(fileobj)
    summary = custom_prompt_summary(text, PROMPT_TEMPLATE, chain_type='map_reduce')
    return summary

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    summary = ''

    if user_input:
        summary = generate_response(user_input)
        
    message(summary)
