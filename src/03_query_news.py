import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Load environment variables
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

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# new version of llama index uses query_engine.query()
query_engine = index.as_query_engine()

# response = query_engine.query("What are some near-term risks to Nvidia?")
# print(response)


response = query_engine.query("Tell me about Google's new supercomputer.")
print(response)





