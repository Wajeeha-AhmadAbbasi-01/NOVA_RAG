import html
import re

import streamlit as st
from langchain_groq import ChatGroq

from rag_engine_agentic import RAGService


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NOVA RAG",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0b0d14;
        color: #e8eaf0;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background: #10131d;
        border-right: 1px solid #242938;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    h1, h2, h3 {
        color: #f4f5f7 !important;
        letter-spacing: -0.02em;
    }

    p {
        color: #aeb4c2;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #303646;
        background: #171b27;
        color: #e9ebf0;
        min-height: 42px;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #6d5dfc;
        background: #1d2030;
        color: #ffffff;
    }

    .stButton > button[kind="primary"] {
        background: #635bff;
        border-color: #635bff;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: #756eff;
        border-color: #756eff;
    }

    [data-testid="stFileUploader"] {
        background: #151924;
        border: 1px dashed #353b4d;
        border-radius: 12px;
        padding: 8px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #635bff;
    }

    [data-testid="stMetric"] {
        background: #131722;
        border: 1px solid #252a38;
        border-radius: 14px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #8f96a7 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f5f6fa !important;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
    }

    [data-testid="stChatMessageContent"] {
        background: #151924;
        border: 1px solid #282e3c;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
    }

    [data-testid="stChatInput"] {
        padding-bottom: 1rem;
    }

    [data-testid="stChatInput"] textarea {
        background: #151924 !important;
        color: #f1f2f5 !important;
        border: 1px solid #303646 !important;
        border-radius: 14px !important;
    }

    [data-testid="stExpander"] {
        background: #121620;
        border: 1px solid #252a38;
        border-radius: 12px;
    }

    hr {
        border-color: #242938 !important;
    }

    .brand-small {
        color: #7770ff;
        font-weight: 700;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
    }

    .hero-label {
        color: #7770ff;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        color: #f5f6fa;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.7rem;
    }

    .hero-description {
        color: #969dad;
        font-size: 1.05rem;
        max-width: 760px;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {

    # Uploaded files
    "uploaded_files": [],

    # One RAG service per document
    "document_kbs": {},

    # Currently selected document
    "active_document": None,

    # Chat history
    "chat_history": [],

    # Conversation summary
    "conversation_summary": "",

    # Suggestions
    "suggestions": [],

    # Current document settings
    "chunk_size": 800,
    "chunk_overlap": 150,
    "top_k": 4,

    # File selection helper
    "last_active_document": None,
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def get_file_signature(file):

    return (
        file.name,
        getattr(file, "size", 0),
    )


def get_active_file():

    active_name = (
        st.session_state.active_document
    )

    if not active_name:

        return None

    for file in st.session_state.uploaded_files:

        if file.name == active_name:

            return file

    return None


def get_active_kb():

    active_name = (
        st.session_state.active_document
    )

    if not active_name:

        return None

    return st.session_state.document_kbs.get(
        active_name
    )


def clear_chat():

    st.session_state.chat_history = []

    st.session_state.conversation_summary = ""


def switch_document(document_name):

    if (
        st.session_state.active_document
        != document_name
    ):

        st.session_state.active_document = (
            document_name
        )

        clear_chat()

        st.session_state.suggestions = []

        # Load saved settings for this document

        kb = st.session_state.document_kbs.get(
            document_name
        )

        if kb:

            st.session_state.chunk_size = (
                kb["chunk_size"]
            )

            st.session_state.chunk_overlap = (
                kb["chunk_overlap"]
            )

            st.session_state.top_k = (
                kb["top_k"]
            )

        else:

            st.session_state.chunk_size = 800
            st.session_state.chunk_overlap = 150
            st.session_state.top_k = 4


def remove_document(document_name):

    # Remove uploaded file

    st.session_state.uploaded_files = [
        file
        for file in st.session_state.uploaded_files
        if file.name != document_name
    ]

    # Remove its independent knowledge base

    if document_name in st.session_state.document_kbs:

        del st.session_state.document_kbs[
            document_name
        ]

    # If removed document was active

    if (
        st.session_state.active_document
        == document_name
    ):

        st.session_state.active_document = None

        clear_chat()

        st.session_state.suggestions = []

        # Select another document automatically

        if st.session_state.uploaded_files:

            new_document = (
                st.session_state.uploaded_files[0].name
            )

            switch_document(
                new_document
            )


def clean_answer(text):

    if not text:

        return ""

    # Remove accidental HTML from LLM output

    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    return html.unescape(
        text
    ).strip()


# ============================================================
# CONVERSATION
# ============================================================

def get_recent_conversation():

    messages = (
        st.session_state.chat_history[-6:]
    )

    lines = []

    for message in messages:

        role = message.get(
            "role",
            "user",
        ).upper()

        content = message.get(
            "content",
            "",
        )

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(lines)


def get_conversation_context():

    summary = (
        st.session_state.conversation_summary
    )

    recent = get_recent_conversation()

    parts = []

    if summary:

        parts.append(
            "Conversation summary:\n"
            + summary
        )

    if recent:

        parts.append(
            "Recent conversation:\n"
            + recent
        )

    return "\n\n".join(parts)


# ============================================================
# CONTEXTUALIZE QUESTION
# ============================================================

def contextualize_question(question):

    if not st.session_state.chat_history:

        return question

    context = get_conversation_context()

    if not context:

        return question

    try:

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_retries=2,
        )

        prompt = f"""
Rewrite the user's latest question as a standalone
retrieval query.

Conversation:

{context}

Latest question:

{question}

Rules:

- Do not answer the question.
- Resolve references such as "it", "this", "that", "they".
- Keep the query concise.
- Do not invent information.
- Return only the rewritten query.
"""

        response = llm.invoke(
            prompt
        )

        rewritten = (
            response.content.strip()
        )

        if rewritten:

            return rewritten

    except Exception:

        pass

    return question


# ============================================================
# CONVERSATION SUMMARY
# ============================================================

def summarize_conversation():

    if (
        len(
            st.session_state.chat_history
        )
        < 8
    ):

        return

    old_messages = (
        st.session_state.chat_history[:-4]
    )

    if not old_messages:

        return

    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in old_messages
    )

    try:

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_retries=2,
        )

        prompt = f"""
Summarize this conversation for future follow-up
questions.

Keep:

- important topics
- important facts
- user goals
- decisions
- unresolved questions

Do not invent information.

Conversation:

{conversation_text}

Return a concise summary under 250 words.
"""

        response = llm.invoke(
            prompt
        )

        st.session_state.conversation_summary = (
            response.content.strip()
        )

        st.session_state.chat_history = (
            st.session_state.chat_history[-4:]
        )

    except Exception:

        pass


# ============================================================
# GENERATE SUGGESTIONS
# ============================================================

def generate_suggestions():

    kb = get_active_kb()

    if not kb:

        return []

    rag = kb.get(
        "rag"
    )

    if rag is None:

        return []

    try:

        documents = rag.documents

        if not documents:

            return []

        selected = documents[:6]

        combined_text = "\n\n".join(
            doc.page_content[:2500]
            for doc in selected
        )

        combined_text = (
            combined_text[:12000]
        )

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_retries=2,
        )

        prompt = f"""
Create exactly 5 useful questions that a user could ask
about this document.

Rules:

- Questions must be answerable from the document.
- Do not invent topics.
- Make the questions different.
- Return one question per line.
- Do not number them.

DOCUMENT:

{combined_text}
"""

        response = llm.invoke(
            prompt
        )

        questions = []

        for line in (
            response.content.splitlines()
        ):

            line = line.strip()

            line = re.sub(
                r"^[\-\*\d\.\)\s]+",
                "",
                line,
            )

            if len(line) > 10:

                questions.append(
                    line
                )

        return questions[:5]

    except Exception:

        return []


# ============================================================
# BUILD ACTIVE DOCUMENT KNOWLEDGE BASE
# ============================================================

def rebuild_active_document():

    active_file = get_active_file()

    if active_file is None:

        st.warning(
            "Select a document first."
        )

        return

    if (
        st.session_state.chunk_overlap
        >= st.session_state.chunk_size
    ):

        st.error(
            "Chunk overlap must be smaller than chunk size."
        )

        return

    document_name = active_file.name

    try:

        with st.status(
            f"Building knowledge base for {document_name}...",
            expanded=True,
        ) as status:

            st.write(
                "Loading document..."
            )

            rag = RAGService(
                chunk_size=(
                    st.session_state.chunk_size
                ),
                chunk_overlap=(
                    st.session_state.chunk_overlap
                ),
                top_k=(
                    st.session_state.top_k
                ),
            )

            st.write(
                "Creating document chunks..."
            )

            info = rag.build_index(
                [active_file]
            )

            st.write(
                "Creating vector index..."
            )

            # Save EVERYTHING specifically
            # for this document.

            st.session_state.document_kbs[
                document_name
            ] = {

                "rag": rag,

                "chunk_size": (
                    st.session_state.chunk_size
                ),

                "chunk_overlap": (
                    st.session_state.chunk_overlap
                ),

                "top_k": (
                    st.session_state.top_k
                ),

                "index_info": info,

                "file_signature": (
                    get_file_signature(
                        active_file
                    )
                ),
            }

            st.session_state.active_document = (
                document_name
            )

            st.session_state.suggestions = (
                generate_suggestions()
            )

            clear_chat()

            status.update(
                label=(
                    f"{document_name} is ready"
                ),
                state="complete",
            )

        st.success(
            f"{document_name} indexed successfully."
        )

    except Exception as e:

        st.error(
            "Could not build the knowledge base."
        )

        st.exception(e)


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(question):

    question = question.strip()

    if not question:

        return

    kb = get_active_kb()

    if not kb:

        st.warning(
            "Build the knowledge base for the selected document first."
        )

        return

    rag = kb.get(
        "rag"
    )

    if rag is None:

        st.warning(
            "Build the knowledge base for this document first."
        )

        return

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:

        retrieval_question = (
            contextualize_question(
                question
            )
        )

        answer, retrieved_docs, agent_activity = (
            rag.agentic_ask(
                retrieval_question,
                conversation_context=(
                    get_conversation_context()
                ),
                max_iterations=2,
            )
        )

        answer = clean_answer(
            answer
        )

        sources = []

        for doc in retrieved_docs:

            metadata = (
                doc.metadata or {}
            )

            source = metadata.get(
                "source",
                "Unknown document",
            )

            page = metadata.get(
                "page",
                None,
            )

            relevance_score = metadata.get(
                "relevance_score",
                None,
            )

            chunk_id = metadata.get(
                "chunk_id",
                "",
            )

            sources.append(
                {
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_id,
                    "relevance_score": relevance_score,
                }
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "agent_activity": agent_activity,
            }
        )

        summarize_conversation()

    except Exception as e:

        if (
            st.session_state.chat_history
            and
            st.session_state.chat_history[-1][
                "role"
            ]
            == "user"
        ):

            st.session_state.chat_history.pop()

        st.error(
            "Something went wrong while generating the answer."
        )

        st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand-small">N O V A &nbsp; R A G</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Document Intelligence"
    )

    st.caption(
        "Upload documents, create separate knowledge bases, "
        "and ask questions about each document independently."
    )

    st.divider()

    # ========================================================
    # DOCUMENT UPLOAD
    # ========================================================

    st.markdown(
        "#### Knowledge Base"
    )

    new_files = st.file_uploader(
        "Add documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload PDF or TXT files.",
    )

    if new_files:

        existing = {
            get_file_signature(file)
            for file in (
                st.session_state.uploaded_files
            )
        }

        added = False

        for file in new_files:

            signature = (
                get_file_signature(file)
            )

            if signature not in existing:

                st.session_state.uploaded_files.append(
                    file
                )

                existing.add(
                    signature
                )

                added = True

        if added:

            # If this is the first document,
            # automatically select it.

            if (
                st.session_state.active_document
                is None
                and st.session_state.uploaded_files
            ):

                first_document = (
                    st.session_state.uploaded_files[
                        0
                    ].name
                )

                switch_document(
                    first_document
                )

            st.rerun()

    # ========================================================
    # DOCUMENT SELECTOR
    # ========================================================

    if st.session_state.uploaded_files:

        document_names = [
            file.name
            for file in (
                st.session_state.uploaded_files
            )
        ]

        # Make sure active document still exists

        if (
            st.session_state.active_document
            not in document_names
        ):

            switch_document(
                document_names[0]
            )

        st.caption(
            "Select knowledge base"
        )

        selected_document = st.selectbox(
            "Active document",
            document_names,
            index=document_names.index(
                st.session_state.active_document
            ),
            label_visibility="collapsed",
        )

        if (
            selected_document
            != st.session_state.active_document
        ):

            switch_document(
                selected_document
            )

            st.rerun()

        # ====================================================
        # DOCUMENT LIST
        # ====================================================

        st.caption(
            f"{len(document_names)} document(s)"
        )

        for index, file in enumerate(
            st.session_state.uploaded_files
        ):

            document_name = file.name

            is_active = (
                document_name
                == st.session_state.active_document
            )

            kb = st.session_state.document_kbs.get(
                document_name
            )

            col1, col2 = st.columns(
                [5, 1],
                vertical_alignment="center",
            )

            with col1:

                if is_active:

                    st.write(
                        f"● 📄 {document_name}"
                    )

                else:

                    st.write(
                        f"○ 📄 {document_name}"
                    )

                if kb:

                    st.caption(
                        "Indexed"
                    )

                else:

                    st.caption(
                        "Not indexed"
                    )

            with col2:

                if st.button(
                    "×",
                    key=f"remove_{document_name}_{index}",
                    help=f"Remove {document_name}",
                ):

                    remove_document(
                        document_name
                    )

                    st.rerun()

    else:

        st.info(
            "No documents uploaded yet."
        )

    st.divider()

    # ========================================================
    # ACTIVE DOCUMENT CONFIGURATION
    # ========================================================

    active_file = get_active_file()

    if active_file:

        st.markdown(
            "#### Active Knowledge Base"
        )

        st.caption(
            active_file.name
        )

        # Load saved settings for this document

        active_kb = get_active_kb()

        if active_kb:

            saved_chunk_size = active_kb[
                "chunk_size"
            ]

            saved_chunk_overlap = active_kb[
                "chunk_overlap"
            ]

            saved_top_k = active_kb[
                "top_k"
            ]

        else:

            saved_chunk_size = (
                st.session_state.chunk_size
            )

            saved_chunk_overlap = (
                st.session_state.chunk_overlap
            )

            saved_top_k = (
                st.session_state.top_k
            )

        # ----------------------------------------------------
        # Chunk size
        # ----------------------------------------------------

        new_chunk_size = st.slider(
            "Chunk size",
            min_value=300,
            max_value=2000,
            value=saved_chunk_size,
            step=50,
            help=(
                "Number of characters in each chunk."
            ),
        )

        # ----------------------------------------------------
        # Chunk overlap
        # ----------------------------------------------------

        new_chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            value=min(
                saved_chunk_overlap,
                new_chunk_size - 1,
            ),
            step=25,
            help=(
                "Characters shared between neighboring chunks."
            ),
        )

        # ----------------------------------------------------
        # Top K
        # ----------------------------------------------------

        new_top_k = st.slider(
            "Retrieved chunks",
            min_value=1,
            max_value=10,
            value=saved_top_k,
            step=1,
            help=(
                "Number of relevant chunks retrieved."
            ),
        )

        # Update current session values

        st.session_state.chunk_size = (
            new_chunk_size
        )

        st.session_state.chunk_overlap = (
            new_chunk_overlap
        )

        st.session_state.top_k = (
            new_top_k
        )

        # ----------------------------------------------------
        # Configuration status
        # ----------------------------------------------------

        st.caption(
            f"Configuration: "
            f"{new_chunk_size} / "
            f"{new_chunk_overlap} / "
            f"Top {new_top_k}"
        )

        # ----------------------------------------------------
        # Rebuild
        # ----------------------------------------------------

        if st.button(
            "Rebuild This Knowledge Base",
            type="primary",
            use_container_width=True,
        ):

            rebuild_active_document()

            st.rerun()

    st.divider()

    # ========================================================
    # CHAT CONTROLS
    # ========================================================

    st.markdown(
        "#### Conversation"
    )

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        clear_chat()

        st.rerun()

    if st.button(
        "Reset workspace",
        use_container_width=True,
    ):

        st.session_state.uploaded_files = []

        st.session_state.document_kbs = {}

        st.session_state.active_document = None

        st.session_state.chat_history = []

        st.session_state.conversation_summary = ""

        st.session_state.suggestions = []

        st.rerun()

    # ========================================================
    # STATUS
    # ========================================================

    st.divider()

    st.markdown(
        "#### System Status"
    )

    active_kb = get_active_kb()

    if active_kb:

        st.success(
            f"{st.session_state.active_document} ready"
        )

    elif active_file:

        st.warning(
            "Knowledge base not built"
        )

    else:

        st.info(
            "Waiting for documents"
        )

    # ========================================================
    # CONVERSATION MEMORY
    # ========================================================

    if st.session_state.conversation_summary:

        st.divider()

        with st.expander(
            "Conversation memory"
        ):

            st.write(
                st.session_state.conversation_summary
            )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="hero-label">AI DOCUMENT INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">NOVA RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-description">
    Ask questions about your documents and get grounded answers
    using semantic retrieval and large language models.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# ACTIVE KNOWLEDGE BASE
# ============================================================

active_kb = get_active_kb()

active_info = None

if active_kb:

    active_info = active_kb.get(
        "index_info"
    )


# ============================================================
# METRICS
# ============================================================

if active_info:

    files_count = active_info.get(
        "files",
        1,
    )

    pages_count = active_info.get(
        "pages",
        0,
    )

    chunks_count = active_info.get(
        "chunks",
        0,
    )

    embedding_dimension = active_info.get(
        "embedding_dimension",
        0,
    )

else:

    files_count = (
        1
        if active_file
        else 0
    )

    pages_count = 0

    chunks_count = 0

    embedding_dimension = 0


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Documents",
        files_count,
    )

with col2:

    st.metric(
        "Pages",
        pages_count,
    )

with col3:

    st.metric(
        "Chunks",
        chunks_count,
    )

with col4:

    st.metric(
        "Vector dimension",
        embedding_dimension or "—",
    )


st.divider()


# ============================================================
# KNOWLEDGE BASE DETAILS
# ============================================================

if active_info:

    with st.expander(
        "Knowledge Base Details"
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Active document**"
            )

            st.code(
                st.session_state.active_document
            )

            st.write(
                "**Embedding model**"
            )

            st.code(
                active_info.get(
                    "embedding_model",
                    "Unknown",
                )
            )

            st.write(
                "**LLM model**"
            )

            st.code(
                active_info.get(
                    "llm_model",
                    "Unknown",
                )
            )

        with col2:

            st.write(
                "**Chunk size**"
            )

            st.write(
                active_kb[
                    "chunk_size"
                ]
            )

            st.write(
                "**Chunk overlap**"
            )

            st.write(
                active_kb[
                    "chunk_overlap"
                ]
            )

            st.write(
                "**Retrieved chunks**"
            )

            st.write(
                active_kb[
                    "top_k"
                ]
            )


# ============================================================
# KNOWLEDGE EXPLORER
# ============================================================

if active_kb:

    rag = active_kb.get(
        "rag"
    )

    if rag is not None:

        with st.expander(
            "Knowledge Explorer"
        ):

            st.caption(
                "Inspect chunks belonging only to the selected document."
            )

            search_text = st.text_input(
                "Search chunks",
                placeholder=(
                    "Search for a word or phrase..."
                ),
                key="chunk_search",
            )

            chunks = rag.chunks

            if search_text:

                search_lower = (
                    search_text.lower()
                )

                matching_chunks = [
                    chunk
                    for chunk in chunks
                    if (
                        search_lower
                        in chunk.page_content.lower()
                    )
                ]

            else:

                matching_chunks = chunks

            st.caption(
                f"Showing "
                f"{min(len(matching_chunks), 50)} "
                f"of "
                f"{len(matching_chunks)} "
                f"matching chunks"
            )

            for i, chunk in enumerate(
                matching_chunks[:50]
            ):

                metadata = (
                    chunk.metadata or {}
                )

                source = metadata.get(
                    "source",
                    "Unknown",
                )

                page = metadata.get(
                    "page",
                    "—",
                )

                chunk_id = metadata.get(
                    "chunk_id",
                    f"chunk-{i}",
                )

                with st.expander(
                    f"{source}  •  Page {page}  •  {chunk_id}"
                ):

                    st.write(
                        chunk.page_content
                    )


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.chat_history:

    if not active_kb:

        if active_file:

            st.markdown(
                "### Build this knowledge base"
            )

            st.write(
                f"**{active_file.name}** is selected. "
                "Configure the retrieval settings in the sidebar "
                "and build its knowledge base."
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.info(
                    "**01 — Select**\n\n"
                    "Choose the document you want to work with."
                )

            with col2:

                st.info(
                    "**02 — Configure**\n\n"
                    "Choose chunk size, overlap and Top-K."
                )

            with col3:

                st.info(
                    "**03 — Build**\n\n"
                    "Create an independent knowledge base."
                )

        else:

            st.markdown(
                "### Start with your documents"
            )

            st.write(
                "Upload one or more PDF or TXT files "
                "from the sidebar."
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.info(
                    "**01 — Upload**\n\n"
                    "Add your documents from the sidebar."
                )

            with col2:

                st.info(
                    "**02 — Select**\n\n"
                    "Choose which document you want to work with."
                )

            with col3:

                st.info(
                    "**03 — Ask**\n\n"
                    "Ask questions about the selected document."
                )

    else:

        st.markdown(
            "### What would you like to know?"
        )

        st.write(
            f"Ask questions about "
            f"**{st.session_state.active_document}**."
        )

        if st.session_state.suggestions:

            columns = st.columns(2)

            for i, suggestion in enumerate(
                st.session_state.suggestions
            ):

                with columns[i % 2]:

                    if st.button(
                        suggestion,
                        key=f"suggestion_{i}",
                        use_container_width=True,
                    ):

                        ask_question(
                            suggestion
                        )

                        st.rerun()

        else:

            st.caption(
                "Ask your own question below."
            )


# ============================================================
# CHAT HISTORY
# ============================================================

for index, message in enumerate(
    st.session_state.chat_history
):

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    if role == "user":

        with st.chat_message(
            "user"
        ):

            st.markdown(
                content
            )

    else:

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                content
            )

            agent_activity = message.get(
                "agent_activity",
                [],
            )

            if agent_activity:

                with st.expander(
                    "🤖 Agent Simulation — How the RAG Agent Decided",
                    expanded=True,
                ):

                    st.caption(
                        "Visible simulation of the agent's actions and decisions. "
                        "Private chain-of-thought is not exposed; only concise, evidence-based reasoning is shown."
                    )

                    for step_number, step in enumerate(
                        agent_activity,
                        start=1,
                    ):

                        if isinstance(step, dict):

                            stage = step.get("stage", f"Step {step_number}")
                            action = step.get("action", "Agent action")
                            detail = step.get("detail", "")
                            rationale = step.get("rationale", "")

                            st.markdown(
                                f"### {stage}"
                            )
                            st.markdown(
                                f"**Agent action:** {action}"
                            )

                            if detail:
                                st.markdown(
                                    f"**What happened:** {detail}"
                                )

                            if rationale:
                                st.markdown(
                                    f"**Why this decision:** {rationale}"
                                )

                            st.divider()

                        else:
                            st.markdown(
                                f"**{step_number}.** {step}"
                            )

            sources = message.get(
                "sources",
                [],
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Sources are now only stored when the RAG
            # service considers the chunks relevant.
            #
            # Therefore an unrelated question will have:
            #
            # sources = []
            #
            # and nothing will be displayed.
            # ------------------------------------------------

            if sources:

                unique_sources = []

                seen = set()

                for source in sources:

                    key = (
                        source.get(
                            "source"
                        ),
                        source.get(
                            "page"
                        ),
                    )

                    if key not in seen:

                        seen.add(
                            key
                        )

                        unique_sources.append(
                            source
                        )

                with st.expander(
                    f"Retrieved evidence "
                    f"({len(unique_sources)})"
                ):

                    for source in unique_sources:

                        source_name = source.get(
                            "source",
                            "Unknown",
                        )

                        page = source.get(
                            "page",
                            "—",
                        )

                        score = source.get(
                            "relevance_score"
                        )

                        if score is not None:

                            st.write(
                                f"📄 **{source_name}** "
                                f"— Page {page} "
                                f"— Relevance {score:.2f}"
                            )

                        else:

                            st.write(
                                f"📄 **{source_name}** "
                                f"— Page {page}"
                            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your selected document..."
)

if question:

    ask_question(
        question
    )

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NOVA RAG • Retrieval-Augmented Document Intelligence"
)