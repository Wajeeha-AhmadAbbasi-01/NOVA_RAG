# ✦ NOVA RAG — Agentic Document Intelligence System

NOVA RAG is an **Agentic Retrieval-Augmented Generation (RAG)** application that allows users to upload documents and ask questions about them using natural language.

The system combines **document retrieval, vector search, LLM reasoning, and grounded response generation** to answer questions using information from the user's uploaded documents.

The main goal of this project is not simply to build a chatbot, but to demonstrate **how an AI system retrieves relevant evidence, reasons over that evidence, and produces a grounded answer**.

---

## 🚀 Key Features

* 📄 Upload PDF and TXT documents
* 🧠 Agentic RAG question answering
* 🔍 Semantic vector search
* 📚 Independent knowledge base for every document
* ⚙️ Per-document chunk size configuration
* ⚙️ Per-document chunk overlap configuration
* 🔢 Per-document Top-K retrieval
* 🧩 Automatic document chunking
* 🗂️ Chroma vector database
* 🤗 HuggingFace sentence-transformer embeddings
* ⚡ Groq LLM inference
* 💬 Conversational chat interface
* 📌 Retrieved evidence shown with answers
* 🚫 Grounded fallback when information is not found
* 💡 Automatically generated suggested questions
* 🔄 Rebuild individual document knowledge bases
* 🗑️ Remove documents independently
* 🎨 Professional Streamlit interface

---

# 🧠 What Makes NOVA RAG Agentic?

Traditional RAG usually follows a simple pipeline:

```text
Question
   ↓
Retrieve Documents
   ↓
Send Context to LLM
   ↓
Generate Answer
```

NOVA RAG extends this idea by making the system explicitly manage the steps involved in answering a question:

```text
                 USER QUESTION
                       │
                       ▼
              ┌─────────────────┐
              │ Question Input  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Active Document │
              │    Selection    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Semantic Search │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Retrieve Top-K  │
              │ Relevant Chunks │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Relevance Check │
              └───────┬─┬───────┘
                      │ │
                Relevant │ Not Relevant
                      │ │
                      ▼ ▼
             ┌──────────┐ ┌────────────────────┐
             │   LLM    │ │ Grounded Fallback  │
             │ Reasoning│ │                    │
             └────┬─────┘ └────────────────────┘
                  │
                  ▼
          ┌───────────────────┐
          │ Grounded Answer   │
          │ + Retrieved       │
          │   Evidence        │
          └───────────────────┘
```

The system does not simply generate an answer from the model's general knowledge.

It first searches the selected document's knowledge base and uses the retrieved evidence as the basis for its answer.

---

# 🏗️ System Architecture

```text
                         NOVA RAG
                            │
                 ┌──────────┴──────────┐
                 │                     │
            Streamlit UI          RAG Service
                 │                     │
                 │             ┌───────┴────────┐
                 │             │                │
                 │        Document Loader    LLM
                 │             │                │
                 │             ▼                ▼
                 │          Chunking          Groq
                 │             │
                 │             ▼
                 │        Embeddings
                 │             │
                 │             ▼
                 │          Chroma
                 │             │
                 └─────────────┘
```

---

# 📚 Multi-Document Architecture

A major design feature of NOVA RAG is that **documents are not merged into one global knowledge base**.

Each uploaded document receives its own independent RAG instance.

For example:

```text
Document A
   │
   ├── Chunks
   ├── Embeddings
   ├── Chroma Collection
   ├── Chunk Size
   ├── Chunk Overlap
   └── Top-K
```

and separately:

```text
Document B
   │
   ├── Chunks
   ├── Embeddings
   ├── Chroma Collection
   ├── Chunk Size
   ├── Chunk Overlap
   └── Top-K
```

This means changing the configuration of Document A does not modify Document B.

The user can select a document and ask questions specifically about that document.

---

# 🔄 End-to-End Workflow

## 1. Upload Document

The user uploads a PDF or TXT file.

```text
PDF / TXT
   ↓
Document Loader
```

For PDFs, pages are processed individually.

For TXT files, the text is loaded as a document.

---

## 2. Text Chunking

Large documents are divided into smaller chunks.

Example:

```text
Original Document
        ↓
 ┌───────────────┐
 │ Chunk 1       │
 ├───────────────┤
 │ Chunk 2       │
 ├───────────────┤
 │ Chunk 3       │
 ├───────────────┤
 │ Chunk 4       │
 └───────────────┘
```

Chunking makes semantic retrieval more precise because the system searches smaller pieces of information rather than an entire document.

NOVA RAG allows the user to configure:

* Chunk Size
* Chunk Overlap
* Top-K

---

# 🧩 Chunk Overlap

Chunk overlap allows neighboring chunks to share some text.

For example:

```text
Chunk 1:
A B C D E F

Chunk 2:
E F G H I J
```

Here, `E F` appears in both chunks.

This helps prevent important information from being split between two chunks.

---

# 🤗 Embeddings

NOVA RAG uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts text into numerical vectors.

For example:

```text
"Machine learning is a subset of AI"
                 ↓
        Embedding Model
                 ↓
[0.12, -0.43, 0.81, ...]
```

These vectors represent the semantic meaning of the text.

---

# 🗄️ Chroma Vector Database

The generated embeddings are stored in **Chroma**.

When the user asks a question, the question is also converted into an embedding.

The system then compares the question embedding with document chunk embeddings.

Conceptually:

```text
User Question
      ↓
Question Embedding
      ↓
Vector Similarity Search
      ↓
Most Relevant Chunks
```

---

# 🔍 Retrieval

NOVA RAG retrieves the configured number of chunks using Top-K retrieval.

For example:

```text
Top-K = 4
```

means the system initially searches for the four most relevant chunks.

The retrieved chunks are then checked using a relevance distance threshold.

Lower distance generally indicates greater similarity.

```text
Question
   ↓
Top 4 chunks
   ↓
Relevance check
   ↓
Relevant evidence
```

If sufficiently relevant information is not found, the system returns:

> I couldn't find that information in the uploaded documents.

This prevents the application from automatically answering unrelated questions.

---

# 🧠 LLM Reasoning

Once relevant evidence has been retrieved, the context is sent to the Groq-hosted LLM.

The system instructs the model to:

* Use only the retrieved document context
* Avoid outside knowledge
* Avoid inventing information
* Answer clearly
* Return a grounded fallback if the evidence is insufficient

Conceptually:

```text
Retrieved Context
       +
User Question
       ↓
      LLM
       ↓
Grounded Answer
```

---

# ⚡ LLM

NOVA RAG uses Groq for LLM inference.

Configured model:

```text
openai/gpt-oss-120b
```

The LLM is used for:

* Question answering
* Reasoning over retrieved context
* Generating document-based suggestions

---

# 🛡️ Grounded Responses

One of the important goals of NOVA RAG is reducing hallucination.

The system does not intentionally allow the LLM to answer every question from its general knowledge.

Instead:

```text
Question
   ↓
Retrieve evidence
   ↓
Is useful evidence available?
      /       \
    YES        NO
     ↓          ↓
    LLM       Fallback
     ↓
Grounded Answer
```

If the retrieved context is insufficient, the application returns:

```text
I couldn't find that information in the uploaded documents.
```

No retrieved evidence is displayed for the fallback response.

---

# 📌 Evidence Display

For successful answers, NOVA RAG can display the retrieved evidence.

Example:

```text
Answer

The document states that ...

▼ Retrieved evidence (2)

Source 1
File: company_policy.pdf
Page: 4

Relevant document content...

Source 2
File: company_policy.pdf
Page: 5

Relevant document content...
```

This improves transparency because the user can inspect the information used to generate the answer.

---

# 💡 Suggested Questions

NOVA RAG can generate example questions based on the active document.

For example, after uploading a company report, the system may generate:

```text
What is the main purpose of the report?

What are the key findings?

What challenges are discussed?

What recommendations are provided?
```

These questions are generated from the selected document rather than from a fixed list.

---

# 🖥️ User Interface

The application is built with **Streamlit**.

The interface contains:

### Sidebar

* Document uploader
* Active document selector
* Uploaded document list
* Document removal
* Chunk size
* Chunk overlap
* Top-K
* Knowledge base rebuild

### Main Area

* Knowledge-base statistics
* Document details
* Indexed chunk explorer
* Suggested questions
* Chat interface
* Retrieved evidence

---

# 🗂️ Project Structure

```text
NOVA-RAG/Agentic_RAG
│
├── app_agentic.py
│
├── rag_engine_agentic.py
│
├── requirements-working.txt
│
├── .env
│
├── .gitignore
│
└── README.md
```

---

# 📄 File Responsibilities

## `app_agentic.py`

Responsible for:

* Streamlit interface
* File upload
* Document selection
* Session state
* Per-document configuration
* Chat interface
* Evidence display
* Suggested questions
* Document management

---

## `rag_engine_agentic.py`

Responsible for the RAG pipeline:

* Document loading
* PDF/TXT processing
* Text chunking
* Embedding generation
* Chroma vector store
* Semantic retrieval
* Relevance filtering
* LLM prompting
* Answer generation
* Suggested-question generation

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd NOVA-RAG
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` to GitHub.

Add it to `.gitignore`:

```text
.env
```

---

# 📦 Requirements

The project uses the following major libraries:

```text
streamlit
python-dotenv
pypdf
langchain
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-chroma
langchain-groq
chromadb
sentence-transformers
huggingface-hub
```

---

# ▶️ Running the Application

Run:

```bash
streamlit run app.py
```

Streamlit will start the application locally.

---

# 🧪 Example Usage

### Step 1

Upload a document:

```text
company_report.pdf
```

### Step 2

NOVA RAG creates:

```text
company_report.pdf
       ↓
Document Loader
       ↓
Chunks
       ↓
Embeddings
       ↓
Chroma Vector Store
```

### Step 3

Ask:

```text
What are the main findings of the report?
```

### Step 4

The system:

```text
Question
   ↓
Semantic Retrieval
   ↓
Relevant Chunks
   ↓
Relevance Check
   ↓
LLM
   ↓
Grounded Answer
```

### Step 5

The application displays:

```text
Answer
+
Retrieved Evidence
```

---

# 🚫 Example of Out-of-Scope Question

Suppose the uploaded document is about a company.

The user asks:

```text
What is the capital of Japan?
```

If the document does not contain this information, NOVA RAG should respond:

```text
I couldn't find that information in the uploaded documents.
```

This demonstrates the grounding behavior of the system.

---

# 🎯 Design Goals

NOVA RAG was designed around five main principles:

### 1. Grounding

Answers should be based on retrieved document evidence.

### 2. Transparency

Retrieved evidence should be visible to the user.

### 3. Document Isolation

Each document maintains its own knowledge base.

### 4. Configurability

Users can adjust:

```text
Chunk Size
Chunk Overlap
Top-K
```

independently for each document.

### 5. Hallucination Reduction

The system uses retrieval and relevance filtering before generating an answer.

---

# 🔬 RAG vs Traditional Chatbot

| Traditional Chatbot            | NOVA RAG                             |
| ------------------------------ | ------------------------------------ |
| General model knowledge        | Document-grounded knowledge          |
| May answer unrelated questions | Relevance filtering                  |
| No document-specific retrieval | Semantic retrieval                   |
| Limited transparency           | Retrieved evidence                   |
| Single general context         | Independent document knowledge bases |
| Harder to inspect sources      | Source/page information available    |

---

# 🤖 Agentic RAG Workflow

The core workflow can be summarized as:

```text
┌──────────────────────┐
│      User Query      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Active Document      │
│ Selection             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Query Embedding      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Vector Retrieval     │
│ Top-K Chunks          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Relevance Evaluation │
└───────┬────────┬─────┘
        │        │
      Valid    Invalid
        │        │
        ▼        ▼
┌────────────┐  ┌─────────────────┐
│ LLM        │  │ Grounded        │
│ Reasoning  │  │ Fallback        │
└─────┬──────┘  └─────────────────┘
      │
      ▼
┌──────────────────────┐
│ Final Answer         │
│ + Evidence           │
└──────────────────────┘
```

---

# 📊 Important Configuration

Default configuration:

```python
chunk_size = 800
chunk_overlap = 150
top_k = 4
```

Retrieval relevance threshold:

```python
RELEVANCE_DISTANCE_THRESHOLD = 0.75
```

The threshold is a practical starting point and may require tuning depending on the embedding model, document type, and Chroma distance behavior.

---

# 🔮 Future Improvements

Possible future improvements include:

* Query rewriting
* Multi-query retrieval
* Hybrid keyword + semantic search
* Reranking retrieved chunks
* Conversation-aware retrieval
* Persistent Chroma storage
* Streaming LLM responses
* Agent tool calling
* Web search as an optional tool
* Automatic document summarization
* Evaluation with RAGAS
* Retrieval precision/recall evaluation
* Advanced citation generation
* Document-level access control
* Production deployment
* Authentication and user accounts

---

# 📈 Possible Production Architecture

A production version could evolve into:

```text
                    User
                     │
                     ▼
                Streamlit UI
                     │
                     ▼
              Agent Controller
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    Retriever     Reranker      Tools
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
                   LLM
                     │
                     ▼
              Grounded Answer
                     │
                     ▼
               User + Sources
```

---

# 🧠 Learning Outcomes

This project demonstrates practical understanding of:

* Retrieval-Augmented Generation
* Agentic AI concepts
* Vector databases
* Semantic embeddings
* Document chunking
* Similarity search
* LLM prompting
* Grounded generation
* Hallucination reduction
* Multi-document architectures
* Streamlit application development
* LangChain
* Chroma
* HuggingFace embeddings
* Groq LLM inference

---

# 👨‍💻 Technology Stack

| Technology    | Purpose                   |
| ------------- | ------------------------- |
| Python        | Core programming language |
| Streamlit     | User interface            |
| LangChain     | RAG orchestration         |
| HuggingFace   | Text embeddings           |
| MiniLM        | Embedding model           |
| Chroma        | Vector database           |
| Groq          | LLM inference             |
| PyPDF         | PDF processing            |
| python-dotenv | Environment configuration |

---

# 📌 Project Summary

**NOVA RAG** is an Agentic RAG-based document intelligence application designed to demonstrate how modern AI systems can retrieve relevant information from private documents and reason over that information before generating a response.

The system combines:

```text
Document Processing
        +
Semantic Embeddings
        +
Vector Retrieval
        +
Relevance Filtering
        +
LLM Reasoning
        +
Evidence
```

to provide a transparent and document-grounded question-answering experience.

---

## ⭐ Future Vision

The long-term vision for NOVA RAG is to evolve from a document Q&A application into a more capable **AI document agent** that can decide when to retrieve information, reformulate queries, use multiple tools, verify evidence, and provide traceable answers.

```text
Current NOVA RAG

Question
   ↓
Retrieve
   ↓
Reason
   ↓
Answer


Future NOVA Agent

Question
   ↓
Understand Goal
   ↓
Plan
   ↓
Retrieve / Search / Tools
   ↓
Evaluate Evidence
   ↓
Reason
   ↓
Verify
   ↓
Answer + Sources
```

---
