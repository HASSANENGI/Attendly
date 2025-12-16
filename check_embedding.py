import os
from dotenv import load_dotenv 
from pinecone import Pinecone  # Updated import


load_dotenv()  # Load environment variables from the .env file
pinecone_api_key = os.getenv('PINECONE_API_KEY') 

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Connect to the index
index_name = "face-embeddings"
index = pc.Index(index_name)  # Use pc.Index instead of pinecone.Index

# Get index metadata
index_info = index.describe_index_stats()
print(index_info)
