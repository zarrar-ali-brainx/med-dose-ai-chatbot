import json
import re
from openai import OpenAI
from tqdm import tqdm
import time
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_embedding(text, model="text-embedding-ada-002"):
    try:
        # Truncate text if it's too long
        max_tokens = 8192
        if len(text.split()) > max_tokens:
            print(f"Text too long for embedding, truncating to {max_tokens} tokens.")
            text = ' '.join(text.split()[:max_tokens])
        
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def clean_existing_data():
    """Remove all existing namespaces and their data"""
    print("\nCleaning existing data...")
    stats = index.describe_index_stats()
    
    for namespace in stats.namespaces:
        if namespace:  # Skip empty namespace
            print(f"Deleting namespace: {namespace}")
            index.delete(delete_all=True, namespace=namespace)
            time.sleep(0.5)  # Rate limiting
    
    print("All existing data cleaned!")

def create_disease_embeddings():
    """Create embeddings for all diseases"""
    try:
        with open("diseases.json", "r") as f:
            diseases_data = json.load(f)
        
        print("\nCreating disease embeddings...")
        
        for disease_name, data in tqdm(diseases_data.items()):
            namespace = disease_name
            print(f"\nProcessing disease: {disease_name}")
            
            # Always create at least one vector per disease
            description = data.get('description', "No description available")
            description_embedding = get_embedding(description)
            
            # Create main disease vector (ensures namespace exists)
            if description_embedding:
                index.upsert(
                    vectors=[{
                        'id': f"{disease_name}_main",
                        'values': description_embedding,
                        'metadata': {
                            'disease_name': disease_name,
                            'type': 'disease_main'
                        }
                    }],
                    namespace=namespace
                )
            
            # Process categories if they exist
            if 'categories' in data and data['categories']:
                def process_category(category, path=[]):
                    # Create embedding for category content
                    content = ' '.join(category.get('content', []))
                    if content:
                        cat_embedding = get_embedding(content)
                        current_path = path + [category['name']]
                        
                        if cat_embedding:
                            index.upsert(
                                vectors=[{
                                    'id': f"{disease_name}_{'_'.join(current_path)}",
                                    'values': cat_embedding,
                                    'metadata': {
                                        'disease_name': disease_name,
                                        'category_path': current_path,
                                        'type': 'category'
                                    }
                                }],
                                namespace=namespace
                            )
                    
                    # Process subcategories recursively
                    if 'subcategories' in category:
                        for subcat in category['subcategories']:
                            process_category(subcat, path + [category['name']])
                
                # Start processing root categories
                for category in data['categories']:
                    process_category(category)
            
            time.sleep(0.1)  # Rate limiting
        
        print("\nDisease embeddings creation completed!")
        print(f"Total diseases processed: {len(diseases_data)}")
    
    except Exception as e:
        print(f"Error creating disease embeddings: {e}")
        import traceback
        traceback.print_exc()

def create_medicine_embeddings():
    """Create embeddings for all medicines"""
    try:
        with open("medicines.json", "r") as f:
            medicines_data = json.load(f)
        
        print("\nCreating medicine embeddings...")
        
        for medicine_name, content in tqdm(medicines_data.items()):
            print(f"\nProcessing medicine: {medicine_name}")
            content_embedding = get_embedding(content)
            
            if content_embedding:
                # Debug: Print the embedding and namespace details
                print(f"Embedding for {medicine_name}: {content_embedding[:5]}... (truncated)")
                print(f"Storing in namespace: 'medicines'")
                
                # Create medicine vector in the 'medicines' namespace
                index.upsert(
                    vectors=[{
                        'id': f"{medicine_name}_content",
                        'values': content_embedding,
                        'metadata': {
                            'medicine_name': medicine_name,
                            'type': 'medicine'
                        }
                    }],
                    namespace="medicines"
                )
            else:
                print(f"Failed to create embedding for {medicine_name}. Skipping.")
            
            time.sleep(0.1)  # Rate limiting
        
        print("\nMedicine embeddings creation completed!")
        print(f"Total medicines processed: {len(medicines_data)}")
    
    except Exception as e:
        print(f"Error creating medicine embeddings: {e}")
        import traceback
        traceback.print_exc()

def verify_new_embeddings():
    """Verify the newly created embeddings"""
    print("\nVerifying new embeddings...")
    stats = index.describe_index_stats()
    
    print(f"\nTotal namespaces: {len(stats.namespaces)}")
    for namespace, data in stats.namespaces.items():
        if namespace:  # Skip empty namespace
            print(f"\nNamespace: {namespace}")
            print(f"Vectors: {data['vector_count']}")

if __name__ == "__main__":
    # Ask for confirmation before proceeding
    response = input("This will delete all existing data and create new embeddings. Proceed? (y/n): ")
    
    if response.lower() == 'y':
        # Clean existing data
        clean_existing_data()
        
        # Create new embeddings for diseases
        create_disease_embeddings()
        
        # Create new embeddings for medicines
        create_medicine_embeddings()
        
        # Verify the results
        verify_new_embeddings()
        
        print("\nProcess completed successfully!")
    else:
        print("Operation cancelled.")

# Verify the embeddings
def verify_embeddings():
    stats = index.describe_index_stats()
    print("\nVerification of stored embeddings:")
    print("=" * 50)
    print(f"Total namespaces: {len(stats.namespaces)}")
    
    for namespace, data in stats.namespaces.items():
        if namespace not in ['', 'medicines']:
            print(f"\nNamespace: {namespace}")
            print(f"Vectors: {data['vector_count']}")
            
            # Get all vectors in this namespace
            results = index.query(
                vector=[0] * 1536,
                top_k=data['vector_count'],
                namespace=namespace,
                include_metadata=True
            )
            
            print("Contents:")
            for match in results['matches']:
                if 'category_path' in match['metadata']:
                    path = " > ".join(match['metadata']['category_path'])
                    print(f"- {path}")
                else:
                    print(f"- Main disease description")

verify_embeddings()

# Check the diseases namespaces
def check_disease_namespaces():
    stats = index.describe_index_stats()
    print("\nNamespace Statistics:")
    print("-" * 50)
    
    for namespace, data in stats.namespaces.items():
        print(f"\nNamespace: {namespace}")
        print(f"Total vectors: {data['vector_count']}")

# Check the results
check_disease_namespaces()

# Add this function after the existing check_disease_namespaces()
def detailed_namespace_check():
    stats = index.describe_index_stats()
    namespaces = stats.namespaces
    
    print("\nDetailed Namespace Analysis:")
    print("=" * 50)
    print(f"Total number of namespaces: {len(namespaces)}")
    print("=" * 50)
    
    # Compare with diseases in JSON
    with open("diseases.json", "r") as f:
        diseases_data = json.load(f)
    
    print(f"Number of diseases in JSON file: {len(diseases_data)}")
    print("\nDisease names in JSON:")
    json_diseases = set(diseases_data.keys())
    for d in sorted(json_diseases):
        print(f"- {d}")
    
    print("\nNamespaces in Pinecone:")
    pinecone_namespaces = set(namespaces.keys())
    for n in sorted(pinecone_namespaces):
        print(f"- {n}")
    
    print("\nExtra namespaces (in Pinecone but not in JSON):")
    extra = pinecone_namespaces - json_diseases
    for e in sorted(extra):
        print(f"- {e}")
    
    # Sort namespaces by vector count
    sorted_namespaces = sorted(namespaces.items(), key=lambda x: x[1]['vector_count'], reverse=True)
    
    print("\nDetailed namespace counts:")
    for namespace, data in sorted_namespaces:
        print(f"\nNamespace: {namespace}")
        print(f"Vectors in namespace: {data['vector_count']}")
        print("-" * 30)

# Run the detailed check
detailed_namespace_check() 