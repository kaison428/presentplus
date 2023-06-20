import os
import openai

# parser
import PyPDF2

# LangChain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document

from langchain.chains.summarize import load_summarize_chain

from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

from langchain.chains.question_answering import load_qa_chain

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def get_text_from_pdf(fileobj):

    #create reader variable that will read the pdffileobj
    reader = PyPDF2.PdfReader(fileobj)
    
    #This will store the number of pages of this pdf file
    num_pages = len(reader.pages)
    
    presentation_text = ''
    for i in range(num_pages):
        #create a variable that will select the selected number of pages
        pageobj = reader.pages[i]
        text = pageobj.extract_text()
        presentation_text += '### Page {} {}'.format(i+1, text)

    return presentation_text


def custom_prompt_summary(input, context_text, chain_type='map_reduce'):

    obj_delimiter = '###'
    presentation_delimiter = '$$$'

    # 1. specify model
    llm = OpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
    chat = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")

    # chain 1: summarize context and generate prompt
    template_key_objectives = """Extract the key objectives from the text \
        that is delimited by triple backticks \
        and format them into a list. \
        text: ```{text}```
        """
    
    prompt_template = ChatPromptTemplate.from_template(template_key_objectives)
    key_objectives_prompt = prompt_template.format_messages(
                    text=context_text)
    key_objectives = chat(key_objectives_prompt).content

    # chain 2: create prompt for summarization
    template_summary = """
        Write a concise summary of the following presentation. The presentation is delimited by '$$$' below:\
    
            $$${text}$$$

        """
    template_summary += f"""
        The summary should be relevant to the following objectives that are delimited by :\
        
            {obj_delimiter}{key_objectives}{obj_delimiter}

        The summary should not repeat any of the above objectives and must orginate from the presentation.\
        
        """.format(context_text=key_objectives)
    
    print(template_summary)
    
    summary_prompt_template = PromptTemplate(template=template_summary, input_variables=["text"])

    # chain 3: summarize input
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(input)
    docs = [Document(page_content=t) for t in texts[:]]

    chain = load_summarize_chain(llm
                                , chain_type=chain_type
                                , map_prompt=summary_prompt_template
                                , combine_prompt=summary_prompt_template)
    summary = chain.run(docs)
    
    return summary

def get_recommendation(input, context_text, chain_type='map_reduce'):

    # 1. specify model
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chat = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")

    # chain 1: summarize context and generate prompt
    template_key_objectives = """Extract the key and unique objectives from the text \
        that is delimited by triple backticks \
        and format them into a list. \
        text: ```{text}```
        """
    
    prompt_template = ChatPromptTemplate.from_template(template_key_objectives)
    key_objectives_prompt = prompt_template.format_messages(
                    text=context_text)
    key_objectives = chat(key_objectives_prompt).content

    # chain 2: create prompt for recommendation
    system_message = """
        You are a knowledgeable and friendly mentor for this presentation. You are to provide a concise and crticial recommendation to the presenter.\
        
        The presentation should meet the following key objectives:\
    
            {text}

        The format of the recommendation should be as follows:\
        
            1. ...\
            2. ...\
            3. ...\

        The recommendations should focus on the content, delivery and structure.\
        """.format(text=key_objectives)
    
    human_message = f'''
        What are your recommendations for the following presention?
        
        {input}

    '''

    messages = [
        SystemMessage(
            content=system_message
        ),
        HumanMessage(
            content=human_message
        ),
    ]

    return chat(messages).content

def get_recommendation_from_vector(input, context_text, chain_type='map_reduce'):

    # 1. specify model
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chat = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")

    # chain 1: summarize context and generate prompt
    template_key_objectives = """Extract the key and unique objectives from the text \
        that is delimited by triple backticks \
        and format them into a list. \
        text: ```{text}```
        """
    
    prompt_template = ChatPromptTemplate.from_template(template_key_objectives)
    key_objectives_prompt = prompt_template.format_messages(
                    text=context_text)
    key_objectives = chat(key_objectives_prompt).content

    # chain 2: create vector database of the presentation
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(input)
    docs = [Document(page_content=t) for t in texts[:]]

    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_documents(docs)

    query ='''
        Please list recommendations for the presentation given the followstreing key objectives:\
    
            {text}

        The format of the recommendation should be as follows:\
        
            1. ...\
            2. ...\
            3. ...\

        The recommendations should focus on the content, delivery and structure.\
    '''.format(text=key_objectives)

    response = index.query(query)

    return response