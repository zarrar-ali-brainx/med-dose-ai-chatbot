import json
from openai import OpenAI
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def debug_disease_namespace(disease_name):
    print(f"\nDEBUG: Contents of namespace '{disease_name}':")
    print("=" * 80)
    
    # Get all vectors in the namespace
    stats = index.describe_index_stats()
    vector_count = stats.namespaces.get(disease_name, {'vector_count': 0})['vector_count']
    
    print(f"Total vectors in namespace: {vector_count}")
    
    # Get all vectors
    dummy_vector = [0.0] * 1536
    results = index.query(
        vector=dummy_vector,
        top_k=vector_count,
        namespace=disease_name,
        include_metadata=True
    )
    
    # Print all metadata
    print("\nAll stored vectors:")
    for i, match in enumerate(results['matches'], 1):
        print(f"\n{i}. Vector ID: {match['id']}")
        print("Metadata:")
        for key, value in match['metadata'].items():
            print(f"  {key}: {value}")
        print("-" * 40)

    # Compare with JSON data
    with open("diseases.json", "r") as f:
        diseases_data = json.load(f)
    
    print("\nOriginal JSON data for this disease:")
    print(json.dumps(diseases_data[disease_name], indent=2))

def query_disease(query_text):
    print(f"\nQuerying for: {query_text}")
    print("=" * 50)
    
    # Load diseases data to get exact disease names
    with open("diseases.json", "r") as f:
        diseases_data = json.load(f)
    
    # Find matching disease name
    disease_name = None
    for name in diseases_data.keys():
        if query_text.lower() in name.lower() or name.lower() in query_text.lower():
            disease_name = name
            break
    
    if not disease_name:
        print("No matching disease found")
        return
    
    print(f"\nIdentified Disease: {disease_name}")
    print("-" * 50)
    
    # Get all content from the disease namespace
    stats = index.describe_index_stats()
    vector_count = stats.namespaces.get(disease_name, {'vector_count': 0})['vector_count']
    
    # Get all vectors in this namespace
    dummy_vector = [0.0] * 1536
    results = index.query(
        vector=dummy_vector,
        top_k=vector_count,
        namespace=disease_name,
        include_metadata=True
    )
    
    # Display results organized by type and category path
    print("\nDisease Description:")
    print("-" * 30)
    for match in results['matches']:
        if match['metadata']['type'] == 'disease_main':
            print(match['metadata']['description'])
            break
    
    print("\nCategories and Content:")
    print("-" * 30)
    
    # Group results by category path
    category_results = []
    for match in results['matches']:
        if match['metadata']['type'] == 'category':
            # Check if 'category_path' exists in metadata
            if 'category_path' in match['metadata']:
                category_results.append({
                    'path': " > ".join(match['metadata']['category_path']),
                    'content': match['metadata']['content']
                })
    
    # Sort by category path to maintain hierarchy
    category_results.sort(key=lambda x: x['path'])
    
    for result in category_results:
        print(f"\n{result['path']}:")
        print(f"Content: {result['content']}")
        print("-" * 20)

# Example usage
query = "VAP"  # or any other disease name
query_disease(query) 