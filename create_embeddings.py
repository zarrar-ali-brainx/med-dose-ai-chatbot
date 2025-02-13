import json
from openai import OpenAI
from tqdm import tqdm
import time
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)

def get_query_embedding(query_text):
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def search_medical_recommendations(query_text):
    query_embedding = get_query_embedding(query_text)
    
    # Search in diseases namespace
    disease_results = index.query(
        vector=query_embedding,
        top_k=1,
        namespace='diseases',
        include_metadata=True
    )
    
    if disease_results['matches']:
        disease_info = disease_results['matches'][0]['metadata']['text']
        print(f"\nIdentified Disease Context:\n{disease_info}")
        
        # Search for relevant medicines
        medicine_results = index.query(
            vector=query_embedding,
            top_k=10,
            namespace='medicines',
            include_metadata=True,
            hybrid_config={
                "alpha": 0.5,
                "query": query_text
            }
        )
        
        print("\nRecommended Medications:")
        print("-" * 50)
        for match in medicine_results['matches']:
            print(f"Relevance Score: {match['score']:.4f}")
            print(f"Recommendation: {match['metadata']['text']}")
            print("-" * 50)

def check_namespaces():
    print("\nChecking Namespace Contents:")
    print("-" * 50)
    
    # Check diseases namespace
    disease_stats = index.describe_index_stats(namespace='diseases')
    total_diseases = disease_stats.namespaces.get('diseases', {'vector_count': 0})['vector_count']
    print(f"\nDiseases Namespace:")
    print(f"Total vectors: {total_diseases}")
    
    # Get all disease entries
    disease_samples = index.query(
        vector=[0] * 1536,  # dummy vector
        top_k=total_diseases,  # Get all diseases
        namespace='diseases',
        include_metadata=True
    )
    print("\nAll Disease Entries:")
    print("-" * 50)
    for match in disease_samples['matches']:
        print(f"ID: {match['id']}")
        print(f"Text: {match['metadata']['text']}")
        print("-" * 50)

    # Check medicines namespace count only
    medicine_stats = index.describe_index_stats(namespace='medicines')
    print(f"\nMedicines Namespace:")
    print(f"Total vectors: {medicine_stats.namespaces.get('medicines', {'vector_count': 0})['vector_count']}")

# Check the namespaces
check_namespaces()

# Comment out the query if you only want to check namespaces
# query = "disease HAP category No MRSA age 21 gender male and crcl 35"
# search_medical_recommendations(query)
