InDocX – Smart Document Assistant

InDocX is an AI-powered Smart Document Assistant built using Retrieval-Augmented Generation (RAG). It enables users to upload PDF documents and interact with them through natural language queries. The system combines semantic search using FAISS with Google Gemini models to provide accurate, context-aware responses directly from uploaded documents.

Live Demo

🔗 Streamlit Application: https://indocx-smart-doc-assistant-jqh4vyb9echbjq8bfu6e5l.streamlit.app/


Features
📄 PDF Document Processing
Upload PDF documents through an intuitive interface.
Automatic text extraction from digital PDFs.
OCR fallback support for scanned documents.
💬 Context-Aware Question Answering
Ask questions about uploaded documents.
Retrieval-Augmented Generation (RAG) ensures answers are grounded in document content.
Evidence-based responses with source chunks.
🔒 Strict Document Mode
Answers are generated strictly from the uploaded document.
Prevents the model from introducing external information.
🌐 Flexible Mode
Uses document content as the primary source.
Can supplement responses with general knowledge when necessary.
🃏 Flashcard Generator
Automatically creates study flashcards from important document concepts.
Useful for revision and self-learning.
📝 Quiz Generator
Generates multiple-choice questions from uploaded documents.
Includes answers and explanations for self-assessment.
📋 Structured Summary Generator
Produces concise document summaries.
Extracts key concepts and important points.
Generates possible exam and viva questions.
🔍 Evidence-Based Responses
Displays the document chunks used to generate answers.
Improves transparency and explainability.
System Architecture
PDF Upload
     │
     ▼
Text Extraction (PyPDF2 / OCR)
     │
     ▼
Text Chunking
     │
     ▼
Gemini Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Similarity Search
     │
     ▼
Google Gemini LLM
     │
     ▼
Answer / Flashcards / Quiz / Summary
Technology Stack
Frontend
Streamlit
Backend
Python
AI & Machine Learning
Google Gemini API
LangChain
FAISS Vector Database
PDF Processing
PyPDF2
pdf2image
Tesseract OCR
Supporting Libraries
NumPy
JSON
Python Dotenv
Installation
Clone Repository
git clone https://github.com/bhoomi-1911/InDocX-Smart-Doc-Assistant.git
cd InDocX-Smart-Doc-Assistant
Create Virtual Environment
python -m venv venv

Activate:

Windows:

venv\Scripts\activate

Linux/Mac:

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Environment Configuration

Create a .env file in the project root:

INDOCX_API_KEY=YOUR_GEMINI_API_KEY

The .env file is excluded from Git using .gitignore and should never be uploaded to GitHub.

Running Locally
streamlit run app.py

The application will be available at:

http://localhost:8501
Streamlit Cloud Deployment
Push the repository to GitHub.
Create a new application on Streamlit Community Cloud.
Select the repository and main branch.
Set the main file path:
app.py
Add the following secret in Streamlit Cloud:
INDOCX_API_KEY = "YOUR_GEMINI_API_KEY"
Deploy the application.
Future Enhancements
Multi-document support
Chat history persistence
Citation highlighting inside PDFs
User authentication
Export flashcards and quizzes
Multi-language document support
Cloud database integration
Real-time collaboration features
