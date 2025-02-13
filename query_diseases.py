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
        # Get the number of vectors in the namespace
        stats = index.describe_index_stats()
        namespace_stats = stats.namespaces.get(disease_name, {})
        vector_count = namespace_stats.get('vector_count', 0)
        
        if vector_count == 0:
            print("No vectors found in namespace")
            return
        
        # Query all vectors in the namespace
        results = index.query(
            vector=[0] * 1536,  # Dummy vector to get all results
            top_k=vector_count,
            namespace=disease_name,
            include_metadata=True
        )
        
        # Organize results by type and category path
        main_info = None
        categories = {}
        
        # First pass: collect all data
        for match in results['matches']:
            metadata = match['metadata']
            if metadata['type'] == 'disease_main':
                main_info = metadata
            elif metadata['type'] == 'category':
                path = tuple(metadata['category_path'])
                categories[path] = metadata
        
        # Display main disease information
        if main_info:
            print("\nDisease Information:")
            print("-" * 20)
            print(main_info.get('description', ''))
        
        # Display categories in hierarchical order
        if categories:
            print("\nCategories:")
            print("-" * 20)
            
            # Sort paths by length and content to maintain hierarchy
            sorted_paths = sorted(categories.keys(), key=lambda x: (len(x), x))
            
            current_indent = 0
            for path in sorted_paths:
                # Calculate indent based on path length
                indent = len(path) - 1
                
                # Print category name with proper indentation
                print(f"\n{'  ' * indent}• {path[-1]}")
                
                # Print category content if available
                metadata = categories[path]
                if 'content' in metadata:
                    content = metadata['content']
                    if isinstance(content, list):
                        for item in content:
                            print(f"{'  ' * (indent + 1)}- {item}")
                    else:
                        print(f"{'  ' * (indent + 1)}- {content}")
        
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