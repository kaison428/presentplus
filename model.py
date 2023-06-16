import os
import openai


# parser
import PyPDF2

# LangChain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

from langchain.docstore.document import Document

from langchain.chains.summarize import load_summarize_chain

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), override=True) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

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

def custom_prompt_summary(input, prompt_template, chain_type='map_reduce'):
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    llm = OpenAI(temperature=0)
    text_splitter = CharacterTextSplitter()

    texts = text_splitter.split_text(input)
    docs = [Document(page_content=t) for t in texts[:3]]

    chain = load_summarize_chain(llm, chain_type=chain_type)
    summary = chain.run(docs)
    chain = load_summarize_chain(OpenAI(temperature=0)
                                , chain_type=chain_type
                                , map_prompt=PROMPT
                                , combine_prompt=PROMPT)
    summary = chain.run(docs)
    
    return summary