Laptop Component Knowledge Base with Interactive Query System
This project is an interactive web application built with Streamlit and Flask, designed to help users upload laptop component images and ask questions about them. The app uses Machine Learning and Natural Language Processing (NLP) to provide real-time, context-aware answers about laptop components such as CPU, RAM, Battery, and more.

Features
Image Upload: Upload images of laptop components (e.g., CPU, RAM, Battery) for identification.
Question Answering: Ask specific questions about laptop components, and get dynamic answers based on a knowledge base.
Real-time Queries: Leverages DistilBERT and Sentence-Transformers to process and understand user queries.
Interactive UI: Easy-to-use interface built with Streamlit.
Technologies Used
Streamlit: Frontend web app for user interaction.
Flask: Backend API for handling image uploads and queries.
DistilBERT: Pre-trained model for question answering.
Sentence-Transformers: For embedding-based query matching.
Python: Primary language for implementation.
How to Run
Clone the repository:

bash
Copy
git clone <repo-url>
cd <project-directory>
Install dependencies:

bash
Copy
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
streamlit run app.py
Access the app at http://localhost:8501 in your web browser.

Future Enhancements
Replace the hardcoded component identification with a custom image classification model.
Expand the knowledge base with more detailed component data.
Fine-tune the NLP models for even better accuracy.