import json
from openai import OpenAI
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def query_disease(disease_query, category_path=None):
    # Load diseases data for structure
    with open("diseases.json", "r") as f:
        diseases_data = json.load(f)
    
    # Find matching disease
    disease_name = next((name for name in diseases_data.keys() 
                        if disease_query.lower() in name.lower()), None)
    
    if not disease_name:
        print("No matching disease found")
        return
    
    print(f"\nIdentified Disease: {disease_name}")
    print("-" * 50)
    
    # Get disease data
    disease_data = diseases_data[disease_name]
    
    # Build category map
    def build_category_map(categories):
        category_map = {}
        for cat in categories:
            path = (cat['name'],)
            category_map[path] = cat.get('content', [])
            if 'subcategories' in cat:
                sub_map = build_category_map(cat['subcategories'])
                category_map.update({path + k: v for k,v in sub_map.items()})
        return category_map
    
    category_map = build_category_map(disease_data.get('categories', []))
    
    # Case 1: No category specified - show interactive selection
    if not category_map:
        # No categories available, show main description
        print("\nDescription:")
        print("-" * 20)
        print(disease_data.get('description', 'No description available'))
        return

    # Interactive category selection
    current_path = []
    while True:
        # Get current level categories
        current_categories = []
        for path in category_map.keys():
            if len(path) == len(current_path) + 1 and path[:len(current_path)] == tuple(current_path):
                current_categories.append(path[-1])

        if not current_categories:
            # Reached a terminal category, show content
            content = category_map.get(tuple(current_path), [])
            if content:
                print(f"\n{' > '.join(current_path)} Content:")
                print("-" * 50)
                for item in content:
                    print(f"• {item}")
            else:
                print("\nNo content available at this category level")
            break

        print(f"\nCurrent category path: {' > '.join(current_path) if current_path else 'Root'}")
        print("Available subcategories:")
        for i, cat in enumerate(current_categories, 1):
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
            index = int(choice) - 1
            if 0 <= index < len(current_categories):
                selected = current_categories[index]
        else:
            for cat in current_categories:
                if choice.lower() == cat.lower():
                    selected = cat
                    break
        
        if selected:
            current_path.append(selected)
        else:
            print("Invalid selection. Please try again.")

# Example usage
if __name__ == "__main__":
    while True:
        query = input("\nEnter disease name to search (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        query_disease(query) 