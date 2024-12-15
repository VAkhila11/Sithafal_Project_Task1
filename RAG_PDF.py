import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Step 1: Extract Text from PDF (specific pages 2 and 6)
def extract_text_from_pdf(pdf_path, pages=[2, 6]):
    text = ""
    reader = PdfReader(pdf_path)
    for page_num in pages:
        text += reader.pages[page_num].extract_text() + "\n"
    return text

# Step 2: Chunk Text for Embedding (Chunk text by paragraphs or sentences)
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Step 3: Convert Text to Embeddings Using TF-IDF
def create_tfidf_embeddings(chunks):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks)
    return embeddings, vectorizer

# Step 4: Retrieve and Generate Answers (Based on Query)
def query_tfidf_database(query, chunks, embeddings, vectorizer):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    most_similar_idx = similarities.argmax()
    return chunks[most_similar_idx]

# Step 5: Extract Tabular Data from Page (Tabular data processing)
def extract_table_from_page(pdf_path, page_num=6):
    reader = PdfReader(pdf_path)
    page = reader.pages[page_num]
    text = page.extract_text()
    
    # Assuming the table data is formatted with clear separators like spaces or tabs
    # A simple parsing based on line breaks or spaces
    lines = text.split("\n")
    table_data = []
    
    for line in lines:
        if line.strip():  # If line is not empty
            table_data.append(line.split())  # Split by spaces (you can adjust if needed)
    
    # Convert to DataFrame for easy viewing and analysis
    df = pd.DataFrame(table_data)
    return df

# Main Implementation
if __name__ == "__main__":
    # PDF Path (Replace with the correct path)
    pdf_path = "Tables, Charts, and Graphs with Examples from History, Economics, Education, Psychology, Urban Affairs and Everyday Life - 2017-2018.pdf"

    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF file not found. Please provide a valid path.")

    # Step 1: Extract text from pages 2 and 6
    print("Extracting text from PDF (Pages 2 and 6)...")
    text = extract_text_from_pdf(pdf_path, pages=[1, 5])  # Note: page index starts from 0, so page 2 is 1, page 6 is 5

    # Step 2: Chunk the extracted text
    print("Chunking text for embeddings...")
    chunks = chunk_text(text)

    # Step 3: Create TF-IDF embeddings
    print("Creating TF-IDF embeddings...")
    embeddings, vectorizer = create_tfidf_embeddings(chunks)

    # Step 4: Query the TF-IDF database
    print("Querying the database...")
    query = "What is the unemployment rate for bachelor's degrees?"  # Example query
    response = query_tfidf_database(query, chunks, embeddings, vectorizer)

    # Print the response
    print("Response:")
    print(response)

    # Step 5: Extract Tabular Data from Page 6
    print("Extracting tabular data from page 6...")
    table_data = extract_table_from_page(pdf_path, page_num=5)  # Page 6 is at index 5
    print("Tabular Data from Page 6:")
    print(table_data)
