import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core import StorageContext, load_index_from_storage, ServiceContext, Settings
import streamlit as st
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


load_dotenv()

llm = Groq(
    model = "llama-3.3-70b-versatile",
    temperature = 0
)
Settings.llm = llm

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
    api_key="AIzaSyDeaSFl9ralY7fIc5pkLrlwAtzXhU3ip4w",
)
Settings.embed_model = embed_model

storage_context = StorageContext.from_defaults(persist_dir= "./storage")

index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

st.title("Financial Stock Analysis using LlamaIndex")

st.header("Reports:")

report_type = st.selectbox(
    'What type of report do you want?',
    ('Single Stock Outlook', 'Competitor Analysis')
)

if report_type == 'Single Stock Outlook':
    symbol = st.text_input("Stock Symbol")

    if symbol:
        with st.spinner(f'Generating report for {symbol}...'):
            response = query_engine.query(f"Write a report on the outlook for {symbol}")
            print(type(response))

            st.write(str(response))

if report_type == 'Competitor Analysis':
    symbol1 = st.text_input("Stock Symbol 1")
    symbol2 = st.text_input("Stock Symbol 2")

    if symbol1 and symbol2:
        with st.spinner(f'Generating report for {symbol1} vs. {symbol2}...'):
            response = query_engine.query(f'Write a report on the competition between {symbol1} stock and {symbol2} stock.')

            st.write(str(response))