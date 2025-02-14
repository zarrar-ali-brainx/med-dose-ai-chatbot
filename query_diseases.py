import json
from openai import OpenAI
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def query_disease(query_text):
    print(f"\nQuerying for: {query_text}")
    print("=" * 50)
    
    # Load diseases data just for name matching
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
    
    try:
        # Get all vectors in the namespace
        results = index.query(
            vector=[0]*1536,
            top_k=1000,
            namespace=disease_name,
            include_metadata=True
        )
        
        # Load original disease data
        with open("diseases.json", "r") as f:
            diseases_data = json.load(f)
        disease_data = diseases_data[disease_name]
        
        # Organize results
        print("\nMain Description:")
        print("-" * 20)
        print(disease_data['description'])
        
        # Build category map from original data
        def build_category_map(categories):
            category_map = {}
            for cat in categories:
                path = (cat['name'],)
                category_map[path] = cat.get('content', [])
                if 'subcategories' in cat:
                    sub_map = build_category_map(cat['subcategories'])
                    category_map.update({(cat['name'],) + k: v for k,v in sub_map.items()})
            return category_map
        
        full_category_map = build_category_map(disease_data['categories'])
        
        # Print categories from Pinecone results
        print("\nCategories:")
        print("-" * 20)
        for match in results['matches']:
            if match['metadata']['type'] == 'category':
                path = tuple(match['metadata']['category_path'])
                if path in full_category_map:
                    print(f"\n{' > '.join(path)}:")
                    for item in full_category_map[path]:
                        print(f"  - {item}")
        
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        import traceback
        traceback.print_exc()

# Example usage
if __name__ == "__main__":
    while True:
        query = input("\nEnter disease name to search (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        query_disease(query) 