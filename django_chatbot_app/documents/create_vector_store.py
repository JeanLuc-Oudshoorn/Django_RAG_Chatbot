import os
import re
import faiss
import numpy as np
from PyPDF2 import PdfReader
import json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
    return text.strip()


def chunk_text(text, chunk_size=2500, overlap=200):
    chunks = []
    positions = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        positions.append((start, end))
        start += chunk_size - overlap
    return chunks, positions


def get_document_title(pdf_path):
    # Open the PDF file
    with open(pdf_path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PdfReader(file)

        # Get the first page
        first_page = pdf_reader.pages[0]

        # Extract text from the first page
        text = first_page.extract_text()

        # Split the text into lines
        lines = text.split("\n")

        # Look for a line that starts with a number followed by a space
        for line in lines:
            # Find occurrence of the string "X45" in line. Split line into 2 strings after the string "X45".
            if "X45" in line:
                line = line.split("X45")[1]

            # Check if the line matches the expected format (number followed by a title) and extract the title if it does
            if re.match(r"^\d+", line):
                # Remove the number
                title = re.sub(r"^\d+", "", line).strip()
                return title

        # If no title is found, return None
        return None


def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up paths
    pdf_folder = os.path.join(script_dir, "Renault-Koleos-Workshop-Manual")
    output_folder = os.path.join(script_dir, "output_docs_manuals")
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Process PDFs and generate embeddings
    all_chunks = []
    all_embeddings = []
    all_metadata = []

    for pdf_file in tqdm(os.listdir(pdf_folder)):
        print(f"Processing {pdf_file}")

        # Extract text and document title from the PDF file
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            document_title = get_document_title(pdf_path)
            # append the name of the document
            document_title = f"{document_title} [{pdf_file}]"
            chunks, _ = chunk_text(text)

            for chunk in chunks:
                # Generate embedding using OpenAI
                response = client.embeddings.create(input=chunk, model="text-embedding-3-small")
                embedding = response.data[0].embedding
                all_chunks.append(chunk)
                all_embeddings.append(embedding)

                metadata = {"source": document_title}
                all_metadata.append(metadata)

    # Convert embeddings to numpy array
    embeddings_array = np.array(all_embeddings).astype("float32")

    # Create and train the FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # Save the FAISS index
    faiss.write_index(index, os.path.join(output_folder, "faiss_index.faiss"))

    # Save the documents (chunks) and metadata
    np.save(os.path.join(output_folder, "documents.npy"), np.array(all_chunks))
    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(all_metadata, f)

    print(f"FAISS index, documents, and metadata saved in {output_folder}")


if __name__ == "__main__":
    main()
