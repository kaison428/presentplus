import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from model import *

# Set API Key
import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


# Set session state
def clear_submit():
    st.session_state["submit"] = False

st.set_page_config(page_title="PresentPlus - An LLM-powered Presentation Mentor", page_icon=":bulb:", layout='wide')
# # Add custom CSS to hide the GitHub icon
# hide_github_icon = """
# #GithubIcon {
#   visibility: hidden;
# }
# """
# st.markdown(hide_github_icon, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("PresentPlus - AI Mentor")

# Sidebar contents
with st.sidebar:
    st.title(':bulb: PresentPlus')
    st.markdown('''
    ## About
    This app is an LLM-powered mentor built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/en/latest/)
    
    ''')

    add_vertical_space(5)
    st.write('Made by HYPE AI :book:')

# Layout of input/response containers
input_container = st.container()
context_container = st.container()
colored_header(label='', description='', color_name='blue-30')
button = st.button("Submit")
response_container = st.container()

# Flag
isPDF = False

# User input
## Function for taking user provided PDF as input
def get_file(key):
    uploaded_file = st.file_uploader(f"Upload your {key} file", type='pdf', key=key, on_change=clear_submit)
    return uploaded_file

## Applying the user input box
with input_container:
    user_input = get_file('presentation')

def get_text(prompt):
    input_text = st.text_area(prompt, "", key="input",  on_change=clear_submit)
    return input_text

## Applying the user context box
with context_container:
    context_text = get_text('Type in the competition requirements:')

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(fileobj, context_text):

    # text = get_text_from_pdf(fileobj)

    # text = get_text_from_pdf(fileobj)
    # summary = custom_prompt_summary(text, context_text, chain_type='map_reduce')
    # recommendation = get_recommendation(text, context_text, chain_type='map_reduce')

    summary = ''
    recommendation = custom_prompt_summary_local('','')
    
    return summary, recommendation

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if button or st.session_state.get("submit"):

        if not user_input:
            st.error("Please upload a document!")
        elif not context_text:
            st.error("Please enter a question!")
        else:
            st.session_state["submit"] = True

            with st.spinner('Wait for it...'):
                summary, recommendation = generate_response(user_input, context_text)
                output = f'''
                    Summary:
                        {summary}
                    
                    Recommendations:
                        {recommendation}
                '''
            st.success('Done!')

            message(output)
