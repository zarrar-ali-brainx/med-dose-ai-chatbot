import docx
import json

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def parse_medicine_data(text):
    # Assuming each medicine starts with a name followed by its description
    medicines = {}
    lines = text.split('\n')
    current_medicine = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.isupper():  # Assuming medicine names are in uppercase
            current_medicine = line
            medicines[current_medicine] = ""
        elif current_medicine:
            medicines[current_medicine] += line + " "

    return medicines

def save_medicine_data_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    text = extract_text_from_docx("medicines.docx")
    medicines_data = parse_medicine_data(text)
    save_medicine_data_to_json(medicines_data, "medicines.json")
    print("Medicine data extracted and saved to medicines.json") 