import json
from docx import Document
from collections import defaultdict
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

def get_category_level(line):
    """Extract category level from the line (e.g., 'Category1', 'Category2', etc.)"""
    if not line.strip().startswith('Category'):
        return None
    for i in range(1, 10):  # Support up to Category9
        if line.strip().startswith(f'Category{i}'):
            return i
    return None

def extract_categories(lines, start_idx):
    """Extract hierarchical categories and their content starting from a given index."""
    categories = []
    current_path = []
    current_content = []
    current_level = 0
    i = start_idx
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Check if we've reached the end of categories (next disease or end of document)
        if line.startswith('Disease:'):
            # Add any remaining content to the last category
            if current_content and current_path:
                categories.append({
                    'path': current_path.copy(),
                    'level': current_level,
                    'content': current_content
                })
            break
        
        category_level = get_category_level(line)
        
        if category_level is not None:
            # If we have accumulated content for the previous category, save it
            if current_content and current_path:
                categories.append({
                    'path': current_path.copy(),
                    'level': current_level,
                    'content': current_content
                })
                current_content = []
            
            # Extract category name (everything after "CategoryX:")
            category_name = line.split(':', 1)[1].strip() if ':' in line else ''
            
            # Update path based on category level
            if category_level <= len(current_path):
                current_path = current_path[:category_level-1]
            current_path.append(category_name)
            current_level = category_level
            
        else:
            # If line doesn't start with 'Category', it's content data
            if line:
                current_content.append(line)
        
        i += 1
    
    # Add any remaining content for the last category only if we haven't already added it
    if current_content and current_path and (not categories or 
        categories[-1]['path'] != current_path or 
        categories[-1]['level'] != current_level):
        categories.append({
            'path': current_path.copy(),
            'level': current_level,
            'content': current_content
        })
    
    return categories, i

def preprocess_diseases(input_file, output_file):
    try:
        # Read the Word document
        doc = Document(input_file)
        
        # Convert document to lines for easier processing
        lines = [paragraph.text for paragraph in doc.paragraphs]
        
        # Dictionary to store processed diseases
        diseases_data = {}
        disease_count = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('Disease:'):
                disease_count += 1
                # Extract disease name
                disease_name = line.replace('Disease:', '').strip()
                
                # Initialize disease data structure
                disease_data = {
                    'name': disease_name,
                    'description': '',
                    'categories': []
                }
                
                # Move to next line
                i += 1
                
                # Collect description until we hit categories
                description_lines = []
                while i < len(lines) and not lines[i].strip().startswith('Category'):
                    if lines[i].strip():
                        description_lines.append(lines[i].strip())
                    i += 1
                
                disease_data['description'] = ' '.join(description_lines)
                
                # Extract categories and their content
                categories, new_i = extract_categories(lines, i)
                
                # Convert flat category list to hierarchical structure
                hierarchical_categories = []
                for cat in categories:
                    if cat['level'] == 1:  # Top-level categories
                        category_data = {
                            'name': cat['path'][-1],
                            'subcategories': [],
                            'content': cat['content'] if cat['content'] else []
                        }
                        hierarchical_categories.append(category_data)
                    else:
                        # Find parent category
                        current = hierarchical_categories
                        for level in range(cat['level'] - 1):
                            if level < len(cat['path']) - 1:
                                parent_name = cat['path'][level]
                                found = False
                                for c in current:
                                    if c['name'] == parent_name:
                                        if 'subcategories' not in c:
                                            c['subcategories'] = []
                                        current = c['subcategories']
                                        found = True
                                        break
                                if not found:
                                    new_category = {
                                        'name': parent_name,
                                        'subcategories': [],
                                        'content': []
                                    }
                                    current.append(new_category)
                                    current = new_category['subcategories']
                        
                        # Add the current category
                        current.append({
                            'name': cat['path'][-1],
                            'content': cat['content'] if cat['content'] else []
                        })
                
                disease_data['categories'] = hierarchical_categories
                i = new_i
                
                # Add to diseases dictionary
                diseases_data[disease_name] = disease_data
            else:
                i += 1
        
        # Save processed data to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(diseases_data, f, indent=2, ensure_ascii=False)
        
        print(f"Preprocessing complete. Found {disease_count} diseases.")
        print(f"Processed data saved to {output_file}.")
        
        # Print example of structure for verification
        if diseases_data:
            print("\nExample of first disease structure:")
            first_disease = next(iter(diseases_data.values()))
            print(json.dumps(first_disease, indent=2))
            
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    input_file = "diseases.docx"
    output_file = "diseases.json"
    preprocess_diseases(input_file, output_file) 