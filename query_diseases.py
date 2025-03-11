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

def get_llm_response_from_embedding(embedding, metadata, prompt_type="description"):
    try:
        # Instead of sending the raw embedding, we'll use metadata to guide the LLM
        disease_name = metadata.get('disease_name', 'unknown disease')
        vector_type = metadata.get('type', 'unknown type')
        
        if vector_type == 'disease_main':
            prompt = f"Generate a comprehensive medical description of {disease_name}. Include symptoms, causes, risk factors, and general treatment approaches."
        elif vector_type == 'category':
            category_path = metadata.get('category_path', [])
            category_name = category_path[-1] if category_path else 'unknown category'
            prompt = f"Generate detailed information about {category_name} for {disease_name}. Include relevant medical details, considerations, and implications."
        else:
            prompt = f"Generate medical information related to {disease_name}."
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical expert with comprehensive knowledge of diseases, treatments, and medical conditions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return None

def query_disease(disease_name, categories=None, age=None, gender=None, height=None, weight=None, crcl=None):
    print(f"\nQuerying for disease: {disease_name}")
    print(f"Categories: {categories}, Age: {age}, Gender: {gender}, Height: {height}, Weight: {weight}, CrCl: {crcl}")

    # Fetch all vectors for this disease namespace
    namespace_results = index.query(
        vector=[0] * 1536,  # Dummy vector to get all entries
        top_k=100,
        namespace=disease_name,
        include_metadata=True
    )
    
    if not namespace_results['matches']:
        print("No embeddings found for this disease.")
        return
    
    # Generate content based on metadata and embeddings
    for match in namespace_results['matches']:
        metadata = match['metadata']
        embedding = match['values']
        
        # Use metadata to determine what kind of content to generate
        if metadata.get('type') == 'disease_main':
            print(f"\n=== MAIN DESCRIPTION FOR {disease_name.upper()} ===")
        elif metadata.get('type') == 'category':
            category_path = metadata.get('category_path', [])
            category_name = ' > '.join(category_path) if category_path else 'Unknown Category'
            print(f"\n=== {category_name.upper()} ===")
        else:
            print(f"\n=== ADDITIONAL INFORMATION ===")
        
        # Generate response using metadata and embedding context
        response = get_llm_response_from_embedding(embedding, metadata)
        print(response if response else "Could not generate response.")
        print("-" * 80)

    # Perform similarity search in medicines namespace
    print("\nPerforming similarity search in medicines namespace...")
    disease_embedding = get_embedding(disease_name)
    if not disease_embedding:
        print("Error generating disease embedding for similarity search.")
        return
    
    medicine_results = index.query(
        vector=disease_embedding,
        top_k=5,
        namespace="medicines",
        include_metadata=True
    )
    
    if not medicine_results['matches']:
        print("No similar medicines found.")
        return
    
    print("\n=== RELATED MEDICINES ===")
    for match in medicine_results['matches']:
        medicine_name = match['metadata'].get('medicine_name', 'Unknown Medicine')
        score = match['score']
        print(f"- {medicine_name} (Relevance: {score:.4f})")
        
        # Generate information about this medicine for this disease
        medicine_prompt = f"Provide a brief overview of how {medicine_name} is used in the treatment of {disease_name}, including dosage considerations for a patient with: Age: {age}, Gender: {gender}, Height: {height}cm, Weight: {weight}kg, Creatinine Clearance: {crcl}ml/min."
        
        medicine_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical expert specializing in pharmacology."},
                {"role": "user", "content": medicine_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        print(medicine_response.choices[0].message.content)
        print("-" * 80)

# Example usage
if __name__ == "__main__":
    while True:
        query = input("\nEnter disease name to search (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        # Get additional parameters
        categories = input("Enter specific categories (or press Enter to skip): ")
        age = input("Enter patient age (or press Enter to skip): ")
        gender = input("Enter patient gender (or press Enter to skip): ")
        height = input("Enter patient height in cm (or press Enter to skip): ")
        weight = input("Enter patient weight in kg (or press Enter to skip): ")
        crcl = input("Enter patient creatinine clearance in ml/min (or press Enter to skip): ")
        
        # Convert inputs to appropriate types
        age = int(age) if age.strip() else None
        height = int(height) if height.strip() else None
        weight = int(weight) if weight.strip() else None
        crcl = int(crcl) if crcl.strip() else None
        
        # Call the query function
        query_disease(
            query, 
            categories=categories if categories.strip() else None,
            age=age,
            gender=gender if gender.strip() else None,
            height=height,
            weight=weight,
            crcl=crcl
        ) 