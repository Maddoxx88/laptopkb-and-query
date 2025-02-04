import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import os

# Initialize the question answering pipeline (using a model like BERT or DistilBERT)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Initialize sentence transformer model for embedding-based retrieval
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Dummy knowledge base - replace with actual data
knowledge_base = {
    "CPU": """
    A CPU (Central Processing Unit) is the primary component of a computer that performs most of the processing inside a computer. 
    Modern CPUs are multi-core, and the clock speed is typically between 2.0 GHz to 5.0 GHz.
    The typical power consumption of a CPU ranges from 35W to 125W.
    """,
    "RAM": """
    RAM (Random Access Memory) is a type of volatile memory that provides fast storage and access to data. 
    It is typically used by the CPU to store data that is actively being used or processed.
    A modern laptop typically has between 4GB to 64GB of RAM.
    """,
    "Motherboard": """
    A motherboard is the primary circuit board that connects all components in a computer. 
    It provides the connections for the CPU, RAM, and other peripherals.
    Modern motherboards support Wi-Fi 6, DDR4 memory, and PCIe 4.0.
    """,
    "Battery": """
    A laptop battery is a power source for the laptop when it's not plugged into an outlet. 
    Battery life varies based on usage but typically lasts between 6-12 hours. 
    The battery can be 4-cell or 6-cell depending on the model.
    """
}

# Function to use QA model for dynamic question answering
def answer_question_from_text(question, context):
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']

# Function to use embeddings for similarity-based search in knowledge base
def get_most_relevant_answer(query):
    query_embedding = embedder.encode([query])
    
    # Find the most similar component description using cosine similarity
    similarities = {}
    for component, text in knowledge_base.items():
        component_embedding = embedder.encode([text])
        cosine_sim = np.dot(query_embedding, component_embedding.T) / (np.linalg.norm(query_embedding) * np.linalg.norm(component_embedding))
        similarities[component] = cosine_sim
    
    # Get the component with the highest similarity
    most_relevant_component = max(similarities, key=similarities.get)
    
    return most_relevant_component, knowledge_base[most_relevant_component]

# Streamlit app structure
st.title("Laptop Component Knowledge Base")
st.write("Upload a component image and ask questions about it!")

# Image Upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying component...")

    # Simulate component identification (You would replace this part with a model or process to identify components)
    component = "CPU"  # Hardcoded for simplicity, replace with image processing logic
    st.write(f"Identified component: {component}")

    # Ask a question
    question = st.text_input(f"Ask a question about {component}:")

    if question:
        # Get the answer based on the question
        component_info = knowledge_base.get(component, "")
        answer = answer_question_from_text(question, component_info)
        st.write(f"Answer: {answer}")

else:
    st.write("Upload an image of the laptop component to get started.")

# Allow the user to query the knowledge base directly
query = st.text_input("Or, ask about any component:")

if query:
    component, context = get_most_relevant_answer(query)
    answer = answer_question_from_text(query, context)
    st.write(f"Component: {component}")
    st.write(f"Answer: {answer}")
