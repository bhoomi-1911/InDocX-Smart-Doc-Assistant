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


Chat history persistence
Citation highlighting inside PDFs
User authentication
Export flashcards and quizzes
Multi-language document support
Cloud database integration
Real-time collaboration features
