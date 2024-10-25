import PyPDF2
import re
import spacy

# Carregar o modelo spaCy em português
nlp = spacy.load('pt_core_news_sm')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(text):
    # Remover números de página, cabeçalhos, rodapés
    text = re.sub(r'\n\d+\n', ' ', text)
    text = re.sub(r'^.*?Capítulo.*?\n', '', text, flags=re.MULTILINE)
    
    # Remover URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remover caracteres especiais e pontuação excessiva
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remover espaços em branco extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_portuguese_text(text):
    doc = nlp(text)
    
    # Tokenização, lematização e remoção de stopwords
    processed_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    
    return " ".join(processed_tokens)

def preprocess_files(input_dir, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(input_path)
            output_filename = f"processed_{os.path.splitext(filename)[0]}.txt"
        elif filename.endswith('.txt'):
            with open(input_path, 'r', encoding='utf-8') as file:
                text = file.read()
            output_filename = f"processed_{filename}"
        else:
            continue
        
        cleaned_text = clean_text(text)
        processed_text = preprocess_portuguese_text(cleaned_text)
        
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(processed_text)
        
        print(f"Preprocessed: {filename}")