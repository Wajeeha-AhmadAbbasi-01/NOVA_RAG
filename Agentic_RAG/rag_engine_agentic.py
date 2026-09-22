import io
import os
import uuid
from functools import lru_cache
from typing import Iterable

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

# Minimum relevance score required for a chunk
# to be considered useful.
#
# Higher = stricter retrieval
# Lower = more permissive retrieval
RELEVANCE_THRESHOLD = 0.25


# ============================================================
# EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


# ============================================================
# LOAD UPLOADED DOCUMENTS
# ============================================================

def load_uploaded_documents(
    uploaded_files: Iterable
):

    documents = []

    for uploaded_file in uploaded_files:

        filename = uploaded_file.name

        file_bytes = uploaded_file.getvalue()

        extension = (
            os.path.splitext(filename)[1]
            .lower()
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if extension == ".pdf":

            reader = PdfReader(
                io.BytesIO(file_bytes)
            )

            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):

                text = page.extract_text() or ""

                text = text.strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "file_type": "pdf",
                            "document_name": filename,
                        },
                    )
                )

        # ----------------------------------------------------
        # TXT
        # ----------------------------------------------------

        elif extension == ".txt":

            text = file_bytes.decode(
                "utf-8",
                errors="ignore",
            )

            text = text.strip()

            if text:

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": 1,
                            "file_type": "txt",
                            "document_name": filename,
                        },
                    )
                )

        else:

            raise ValueError(
                f"Unsupported file type: {filename}"
            )

    if not documents:

        raise ValueError(
            "No readable content was found in the uploaded document."
        )

    return documents


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        metadata = document.metadata or {}

        source = metadata.get(
            "source",
            "Unknown document",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        context_parts.append(
            f"""
[Source {index}: {source}, Page {page}]

{document.page_content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# RAG SERVICE
# ============================================================

class RAGService:

    def __init__(
        self,
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
    ):

        # ----------------------------------------------------
        # Validate API key
        # ----------------------------------------------------

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to your .env file."
            )

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if chunk_size <= 0:

            raise ValueError(
                "Chunk size must be greater than zero."
            )

        if chunk_overlap < 0:

            raise ValueError(
                "Chunk overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:

            raise ValueError(
                "Chunk overlap must be smaller than chunk size."
            )

        if top_k <= 0:

            raise ValueError(
                "Top-K must be greater than zero."
            )

        # ----------------------------------------------------
        # Store configuration
        # ----------------------------------------------------

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        self.embedding_model = (
            get_embedding_model()
        )

        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            max_retries=2,
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        self.prompt = ChatPromptTemplate.from_template(
            """
You are NOVA RAG, a document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided document context.

CONTEXT:
{context}

CONVERSATION:
{conversation_context}

QUESTION:
{question}

Rules:

1. Use only the provided document context for factual answers.
2. Do not invent or assume information.
3. If the answer is not present in the context, respond exactly:

I couldn't find that information in the uploaded documents.

4. Conversation history can be used to understand follow-up
   questions, but factual claims must still come from the
   provided document context.
5. Give a clear and concise answer.
"""
        )

        # ----------------------------------------------------
        # Answer chain
        # ----------------------------------------------------

        self.answer_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.vector_store = None
        self.retriever = None

        self.documents = []
        self.chunks = []

        self.collection_name = None

        self.embedding_dimension = None

    # ========================================================
    # BUILD INDEX
    # ========================================================

    def build_index(
        self,
        uploaded_files,
    ):

        # ----------------------------------------------------
        # Load documents
        # ----------------------------------------------------

        documents = load_uploaded_documents(
            uploaded_files
        )

        self.documents = documents

        # ----------------------------------------------------
        # Split documents
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

        chunks = splitter.split_documents(
            documents
        )

        if not chunks:

            raise ValueError(
                "No chunks were created from the document."
            )

        # ----------------------------------------------------
        # Add stable chunk metadata
        # ----------------------------------------------------

        document_name = uploaded_files[0].name

        for index, chunk in enumerate(
            chunks
        ):

            chunk.metadata = {
                **chunk.metadata,

                "document_name": document_name,

                "chunk_id": (
                    f"{document_name}_chunk_{index}"
                ),

                "chunk_index": index,
            }

        self.chunks = chunks

        # ----------------------------------------------------
        # Unique Chroma collection
        # ----------------------------------------------------

        self.collection_name = (
            "nova_rag_"
            + uuid.uuid4().hex
        )

        # ----------------------------------------------------
        # Create vector store
        # ----------------------------------------------------

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
        )

        chunk_ids = [
            chunk.metadata["chunk_id"]
            for chunk in chunks
        ]

        self.vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        # ----------------------------------------------------
        # Retriever
        # ----------------------------------------------------

        self.retriever = (
            self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.top_k,
                },
            )
        )

        # ----------------------------------------------------
        # Embedding dimension
        # ----------------------------------------------------

        try:

            sample_embedding = (
                self.embedding_model.embed_query(
                    "dimension test"
                )
            )

            self.embedding_dimension = len(
                sample_embedding
            )

        except Exception:

            self.embedding_dimension = 0

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        pages = len(documents)

        return {
            "files": 1,
            "pages": pages,
            "chunks": len(chunks),
            "embedding_dimension": (
                self.embedding_dimension
            ),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": GROQ_MODEL,
            "document_name": document_name,
        }

    # ========================================================
    # RETRIEVE RELEVANT DOCUMENTS
    # ========================================================

    def retrieve_relevant_documents(
        self,
        question,
    ):

        if self.vector_store is None:

            raise RuntimeError(
                "Knowledge base has not been built."
            )

        # ----------------------------------------------------
        # Similarity search with scores
        # ----------------------------------------------------

        results = (
            self.vector_store
            .similarity_search_with_relevance_scores(
                question,
                k=self.top_k,
            )
        )

        relevant_documents = []

        for document, score in results:

            # Keep the retrieved chunks instead of rejecting them
            # too aggressively. Chroma's relevance-score scale can
            # vary depending on the vector-store configuration, so a
            # fixed threshold can incorrectly make valid answers look
            # like "no information was found".
            document.metadata = {
                **document.metadata,
                "relevance_score": float(score),
            }

            relevant_documents.append(document)

        # If the relevance-score API returns no usable chunks,
        # fall back to normal similarity search so the RAG pipeline
        # can still use the closest document chunks.
        if not relevant_documents:
            relevant_documents = self.vector_store.similarity_search(
                question,
                k=self.top_k,
            )

        return relevant_documents

    # ========================================================
    # AGENTIC RAG HELPERS
    # ========================================================

    def _agent_llm(self):

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            max_retries=2,
        )

    def _grade_context(self, question, documents):

        context = format_context(documents)

        prompt = f"""
You are the context evaluator in an Agentic RAG system.

Question:
{question}

Retrieved document context:
{context}

Decide whether the retrieved context contains enough
information to answer the question accurately.

Return ONLY one word:
SUFFICIENT
or
INSUFFICIENT

Do not answer the question.
"""

        response = self._agent_llm().invoke(prompt)
        decision = str(response.content).strip().upper()

        if "SUFFICIENT" in decision and "INSUFFICIENT" not in decision:
            return "SUFFICIENT"

        return "INSUFFICIENT"

    def _explain_context_decision(self, question, documents, decision):

        context = format_context(documents) if documents else "No useful context was retrieved."

        prompt = f"""
You are generating a SHORT, user-visible explanation of an Agentic RAG decision.
Do NOT reveal chain-of-thought, hidden reasoning, private deliberation, or internal model thoughts.
Give only a concise evidence-based rationale that a student can show to a teacher.

Question:
{question}

Retrieved evidence:
{context}

Decision:
{decision}

Explain in 1-2 sentences why this retrieval was considered sufficient or insufficient.
Mention the type of information present or missing, without inventing facts.
"""

        response = self._agent_llm().invoke(prompt)
        return str(response.content).strip()

    def _explain_verification(self, question, context, answer, verification):

        prompt = f"""
You are generating a SHORT, user-visible explanation of a RAG grounding check.
Do NOT reveal chain-of-thought, hidden reasoning, private deliberation, or internal model thoughts.

Question:
{question}

Evidence:
{context}

Draft answer:
{answer}

Verification result:
{verification}

Explain in 1-2 sentences what was checked and why the result was PASS or FAIL.
Only discuss observable evidence alignment.
"""

        response = self._agent_llm().invoke(prompt)
        return str(response.content).strip()

    def _rewrite_retrieval_query(self, question, documents):

        context = format_context(documents) if documents else "No useful context was retrieved."

        prompt = f"""
You are the query-refinement step in an Agentic RAG system.

Original question:
{question}

The previous retrieval did not provide enough information.
Previous retrieved context:
{context}

Create a better retrieval query that should find the missing
information in the uploaded document.

Rules:
- Do not answer the question.
- Do not invent facts.
- Include important concepts, names, terms, conditions, or relationships
  from the question.
- Return ONLY the improved retrieval query.
"""

        response = self._agent_llm().invoke(prompt)
        rewritten = str(response.content).strip()

        return rewritten or question

    def agentic_ask(
        self,
        question,
        conversation_context="",
        max_iterations=2,
    ):
        """Run Agentic RAG with a teacher-friendly, high-level simulation trace.

        The trace shows observable agent actions, retrieval decisions, query
        refinement, evidence checks, and verification. It intentionally does
        not expose private chain-of-thought.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.vector_store is None:
            raise RuntimeError("Please build the knowledge base first.")

        trace = []

        trace.append({
            "stage": "1. Analyze",
            "action": "Analyze the user's question",
            "detail": f"I received: {question}",
        })

        retrieval_query = question
        trace.append({
            "stage": "2. Plan retrieval",
            "action": "Create the first retrieval query",
            "detail": retrieval_query,
        })

        relevant_documents = []
        final_decision = "INSUFFICIENT"

        for attempt in range(max_iterations):

            trace.append({
                "stage": f"3.{attempt + 1} Retrieve",
                "action": f"Search the knowledge base (attempt {attempt + 1})",
                "detail": f"Query used: {retrieval_query}",
            })

            relevant_documents = self.retrieve_relevant_documents(
                retrieval_query
            )

            trace.append({
                "stage": f"3.{attempt + 1} Retrieve",
                "action": "Collect the most relevant document chunks",
                "detail": (
                    f"Retrieved {len(relevant_documents)} chunk(s)."
                    if relevant_documents
                    else "No usable chunks were returned."
                ),
            })

            if not relevant_documents:
                decision = "INSUFFICIENT"
                rationale = "No document evidence was available to support an answer."
            else:
                decision = self._grade_context(
                    question,
                    relevant_documents,
                )
                rationale = self._explain_context_decision(
                    question,
                    relevant_documents,
                    decision,
                )

            final_decision = decision

            trace.append({
                "stage": f"4.{attempt + 1} Evaluate context",
                "action": "Check whether the retrieved evidence is enough",
                "detail": f"Decision: {decision}",
                "rationale": rationale,
            })

            if decision == "SUFFICIENT":
                trace.append({
                    "stage": "5. Decide",
                    "action": "Context is sufficient",
                    "detail": "Proceeding to answer generation using the retrieved evidence.",
                })
                break

            if attempt < max_iterations - 1:
                rewritten = self._rewrite_retrieval_query(
                    question,
                    relevant_documents,
                )

                trace.append({
                    "stage": "5. Refine",
                    "action": "Context is insufficient, so refine the retrieval query",
                    "detail": f"New query: {rewritten}",
                })

                retrieval_query = rewritten
            else:
                trace.append({
                    "stage": "5. Decide",
                    "action": "Context remains insufficient",
                    "detail": "No further retrieval attempt is available.",
                })

        if not relevant_documents or final_decision != "SUFFICIENT":
            return (
                "I couldn't find that information in the uploaded documents.",
                [],
                trace,
            )

        trace.append({
            "stage": "6. Generate",
            "action": "Generate the answer from retrieved evidence",
            "detail": "The answer generator is restricted to the retrieved document context.",
        })

        context = format_context(relevant_documents)

        answer = self.answer_chain.invoke(
            {
                "context": context,
                "question": question,
                "conversation_context": conversation_context,
            }
        )

        answer = str(answer).strip()

        fallback = "I couldn't find that information in the uploaded documents."

        if not answer or fallback.lower() in answer.lower():
            trace.append({
                "stage": "6. Generate",
                "action": "No grounded answer could be generated",
                "detail": "Returning the document-not-found response.",
            })
            return fallback, [], trace

        trace.append({
            "stage": "7. Verify",
            "action": "Check whether the draft answer is grounded in the retrieved evidence",
            "detail": "Comparing the answer's claims with the retrieved document context.",
        })

        verify_prompt = f"""
You are the final grounding verifier in a RAG system.

Question:
{question}

Document evidence:
{context}

Draft answer:
{answer}

Does the draft answer stay supported by the document evidence?
Return ONLY one word: PASS or FAIL.
"""

        verification = str(
            self._agent_llm().invoke(verify_prompt).content
        ).strip().upper()

        if "PASS" in verification and "FAIL" not in verification:
            verification_result = "PASS"
        else:
            verification_result = "FAIL"

        verification_rationale = self._explain_verification(
            question,
            context,
            answer,
            verification_result,
        )

        trace.append({
            "stage": "7. Verify",
            "action": f"Grounding check: {verification_result}",
            "detail": verification_rationale,
        })

        if verification_result == "PASS":
            trace.append({
                "stage": "8. Final",
                "action": "Final answer ready",
                "detail": "The answer passed the document-grounding check.",
            })
            return answer, relevant_documents, trace

        trace.append({
            "stage": "8. Final",
            "action": "Grounding check failed",
            "detail": "The draft was not returned as a trusted final answer.",
        })

        return fallback, [], trace

    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask(
        self,
        question,
        conversation_context="",
    ):

        if not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        if self.vector_store is None:

            raise RuntimeError(
                "Please build the knowledge base first."
            )

        # ----------------------------------------------------
        # Retrieve relevant chunks
        # ----------------------------------------------------

        relevant_documents = (
            self.retrieve_relevant_documents(
                question
            )
        )

        # ----------------------------------------------------
        # No relevant information
        # ----------------------------------------------------

        if not relevant_documents:

            return (
                "I couldn't find that information "
                "in the uploaded documents.",
                [],
            )

        # ----------------------------------------------------
        # Format context
        # ----------------------------------------------------

        context = format_context(
            relevant_documents
        )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        answer = self.answer_chain.invoke(
            {
                "context": context,
                "question": question,
                "conversation_context": (
                    conversation_context
                ),
            }
        )

        answer = answer.strip()

        # ----------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------

        if not answer:

            answer = (
                "I couldn't find that information "
                "in the uploaded documents."
            )

            return answer, []

        # ----------------------------------------------------
        # If LLM itself says information was not found,
        # don't expose retrieved chunks as evidence.
        # ----------------------------------------------------

        fallback = (
            "I couldn't find that information "
            "in the uploaded documents."
        )

        if fallback.lower() in answer.lower():

            return fallback, []

        return (
            answer,
            relevant_documents,
        )