import json
from docx import Document
import os

def clear_cache():
    """Clear any existing cache files"""
    if os.path.exists("diseases.json"):
        os.remove("diseases.json")
    print("Cache cleared.")

def extract_categories(lines, start_idx):
    """Extract hierarchical categories and their content starting from a given index."""
    categories = []
    current_path = []
    current_content = []
    current_level = 0
    i = start_idx
    
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
            
        if line.startswith('Disease'):
            break
        
        category_level = None
        if line.startswith('Category'):
            for level in range(1, 10):
                if line.startswith(f'Category{level}'):
                    category_level = level
                    break
        
        if category_level is not None:
            if current_content and current_path:
                categories.append({
                    'path': current_path.copy(),
                    'level': current_level,
                    'content': current_content
                })
                current_content = []
            
            category_name = line.split(':', 1)[1].strip() if ':' in line else line.split(' ', 1)[1].strip()
            
            if category_level <= len(current_path):
                current_path = current_path[:category_level-1]
            current_path.append(category_name)
            current_level = category_level
            
        else:
            if line:
                current_content.append(line)
        
        i += 1
    
    # Add any remaining content
    if current_content and current_path:
        categories.append({
            'path': current_path.copy(),
            'level': current_level,
            'content': current_content
        })
    
    return categories, i

def preprocess_diseases(input_file, output_file):
    try:
        # Clear any existing cache
        clear_cache()
        
        # Read the Word document
        doc = Document(input_file)
        
        # Convert document to lines
        lines = [paragraph.text for paragraph in doc.paragraphs]
        
        # Dictionary to store processed diseases
        diseases_data = {}
        disease_count = 0
        all_diseases = []
        
        print("\nScanning document for diseases...")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if 'Disease' in line:
                disease_count += 1
                # Extract disease name
                disease_name = line.split(':', 1)[1].strip() if ':' in line else line.split('Disease', 1)[1].strip()
                all_diseases.append(disease_name)
                
                print(f"\nProcessing Disease #{disease_count}")
                print(f"Name: {disease_name}")
                
                # Initialize disease data
                disease_data = {
                    'name': disease_name,
                    'description': '',
                    'categories': []
                }
                
                i += 1
                
                # Collect description until next disease or category
                description_lines = []
                while i < len(lines):
                    current_line = lines[i].strip()
                    # Stop if we hit a category or another disease
                    if current_line.startswith('Category') or 'Disease' in current_line:
                        break
                    if current_line:  # Only add non-empty lines
                        description_lines.append(current_line)
                    i += 1
                
                disease_data['description'] = ' '.join(description_lines)
                
                # Only process categories if they exist
                if i < len(lines) and lines[i].strip().startswith('Category'):
                    categories, new_i = extract_categories(lines, i)
                    i = new_i  # Update index only if categories were processed
                    
                    # Convert to hierarchical structure
                    hierarchical_categories = []
                    for cat in categories:
                        current = hierarchical_categories
                        for level, name in enumerate(cat['path']):
                            if level == 0:
                                # Find or create top-level category
                                found = False
                                for existing in hierarchical_categories:
                                    if existing['name'] == name:
                                        current = existing['subcategories']
                                        found = True
                                        break
                                if not found:
                                    new_cat = {'name': name, 'subcategories': [], 'content': []}
                                    hierarchical_categories.append(new_cat)
                                    current = new_cat['subcategories']
                            else:
                                # Find or create subcategory
                                found = False
                                for existing in current:
                                    if existing['name'] == name:
                                        if 'subcategories' not in existing:
                                            existing['subcategories'] = []
                                        current = existing['subcategories']
                                        found = True
                                        break
                                if not found:
                                    new_cat = {'name': name, 'subcategories': [], 'content': []}
                                    current.append(new_cat)
                                    current = new_cat['subcategories']
                        
                        # Add content to the last category in the path
                        if cat['content']:
                            current = hierarchical_categories
                            for name in cat['path'][:-1]:
                                for c in current:
                                    if c['name'] == name:
                                        current = c['subcategories']
                                        break
                            for c in current:
                                if c['name'] == cat['path'][-1]:
                                    c['content'] = cat['content']
                                    break
                    
                    disease_data['categories'] = hierarchical_categories
                
                # Add to diseases dictionary
                diseases_data[disease_name] = disease_data
                
                print(f"Processed disease: {disease_name}")
                if not disease_data['categories']:
                    print(f"Note: No categories found for this disease")
                
            else:
                i += 1
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(diseases_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessing complete!")
        print(f"Total diseases found: {disease_count}")
        print("\nDiseases processed:")
        for idx, disease in enumerate(all_diseases, 1):
            print(f"{idx}. {disease}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    input_file = "diseases.docx"
    output_file = "diseases.json"
    preprocess_diseases(input_file, output_file) 