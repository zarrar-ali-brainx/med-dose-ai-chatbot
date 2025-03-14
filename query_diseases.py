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
    """Get embedding for the query text"""
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def list_available_diseases():
    """List all available diseases from Pinecone namespaces"""
    stats = index.describe_index_stats()
    namespaces = stats.namespaces
    
    print("\nAvailable diseases:")
    print("-" * 50)
    
    # Filter out empty namespace and non-disease namespaces
    disease_namespaces = [ns for ns in namespaces.keys() if ns and ns != 'medicines']
    
    for i, disease in enumerate(sorted(disease_namespaces), 1):
        print(f"{i}. {disease}")
    
    return sorted(disease_namespaces)

def query_disease(disease_query):
    # Get all available diseases
    available_diseases = list_available_diseases()
    
    # Find matching disease
    disease_name = None
    for name in available_diseases:
        if disease_query.lower() in name.lower():
            disease_name = name
            break
    
    if not disease_name:
        print("No matching disease found")
        return
    
    print(f"\nIdentified Disease: {disease_name}")
    print("-" * 50)
    
    # Create a dummy vector for filtering (required by Pinecone)
    dummy_vector = [0.0] * 1536  # Assuming 1536 dimensions for text-embedding-ada-002
    
    # Query the main disease vector
    main_results = index.query(
        vector=dummy_vector,  # Add this dummy vector
        namespace=disease_name,
        filter={"type": "disease_main"},
        top_k=1,
        include_metadata=True
    )
    
    if main_results['matches']:
        print("\nDescription:")
        print("-" * 20)
        # Display the description from metadata
        description = main_results['matches'][0]['metadata'].get('description', 'No description available')
        print(description)
    
    # Get all category vectors for this disease
    category_results = index.query(
        vector=dummy_vector,  # Add this dummy vector
        namespace=disease_name,
        filter={"type": "category"},
        top_k=100,  # Adjust based on expected number of categories
        include_metadata=True
    )
    
    # Build category structure from results
    category_paths = {}
    for match in category_results['matches']:
        if 'category_path' in match['metadata']:
            path = tuple(match['metadata']['category_path'])
            category_paths[path] = match['id']
    
    if not category_paths:
        print("\nNo categories available for this disease")
        return
    
    # Interactive category selection
    current_path = []
    while True:
        # Get current level categories
        current_categories = []
        for path in category_paths.keys():
            if len(path) == len(current_path) + 1 and path[:len(current_path)] == tuple(current_path):
                current_categories.append(path[-1])
        
        if not current_categories:
            # Reached a terminal category, show content
            full_path = tuple(current_path)
            if full_path in category_paths:
                vector_id = category_paths[full_path]
                
                # Query for this specific category
                content_result = index.fetch(
                    ids=[vector_id],
                    namespace=disease_name
                )
                
                # Check if the vector exists in the response
                # Updated to handle the new response format
                if hasattr(content_result, 'vectors') and vector_id in content_result.vectors:
                    print(f"\n{' > '.join(current_path)} Content:")
                    print("-" * 50)
                    # Display the content from metadata
                    content = content_result.vectors[vector_id].metadata.get('content', 'No content available')
                    print(content)
                else:
                    print("\nNo content available for this category")
            else:
                print("\nNo content available at this category level")
            break
        
        print(f"\nCurrent category path: {' > '.join(current_path) if current_path else 'Root'}")
        print("Available subcategories:")
        for i, cat in enumerate(sorted(current_categories), 1):
            print(f"{i}. {cat}")
        print("\nCommands: 'back', 'exit', or select a category number/name")
        
        choice = input("Enter your choice: ").strip()
        
        if choice.lower() == 'exit':
            return
        elif choice.lower() == 'back':
            if current_path:
                current_path.pop()
            continue
        
        # Try to match by number or name
        selected = None
        if choice.isdigit():
            index_num = int(choice) - 1
            sorted_cats = sorted(current_categories)
            if 0 <= index_num < len(sorted_cats):
                selected = sorted_cats[index_num]
        else:
            for cat in current_categories:
                if choice.lower() == cat.lower():
                    selected = cat
                    break
        
        if selected:
            current_path.append(selected)
        else:
            print("Invalid selection. Please try again.")

def semantic_search(query_text):
    """Search across all diseases using semantic search"""
    print(f"\nPerforming semantic search for: '{query_text}'")
    print("-" * 50)
    
    # Get embedding for the query
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print("Failed to generate embedding for the query")
        return
    
    # Search across all namespaces
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    if not results['matches']:
        print("No matching results found")
        return
    
    print("\nTop results:")
    for i, match in enumerate(results['matches'], 1):
        disease = match['metadata'].get('disease_name', 'Unknown')
        score = match['score']
        
        if 'category_path' in match['metadata']:
            category = " > ".join(match['metadata']['category_path'])
            print(f"{i}. Disease: {disease} | Category: {category} | Score: {score:.4f}")
        else:
            print(f"{i}. Disease: {disease} | Main description | Score: {score:.4f}")

# Example usage
if __name__ == "__main__":
    query = input("\nEnter disease name to search: ")
    query_disease(query) 