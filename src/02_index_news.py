import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


# Load environment variables
load_dotenv()

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
    api_key="AIzaSyDeaSFl9ralY7fIc5pkLrlwAtzXhU3ip4w",
)
Settings.embed_model = embed_model

documents = SimpleDirectoryReader('articles').load_data()

index = VectorStoreIndex.from_documents(documents)

# llama index 0.6 replaces index.save_to_disk() with index.storage_context.persist()
# json files will be stored in a storage/ directory instead of index_new.json
# index.save_to_disk('index_news.json')

index.storage_context.persist()