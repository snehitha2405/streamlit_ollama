import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄"
)
st.markdown("""
<style>

/* ==========================================
   MAIN APP
========================================== */

.stApp {
    background: linear-gradient(
        135deg,
        #eef4ff 0%,
        #f8faff 50%,
        #eef6ff 100%
    );
}


/* ==========================================
   MAIN CONTENT
========================================== */

.block-container {
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================
   TITLE
========================================== */

h1 {
    color: #173b67 !important;
    text-align: center;
    font-size: 42px !important;
    font-weight: 750 !important;
    letter-spacing: -1px;
    margin-bottom: 8px;
}


/* ==========================================
   SUBTITLE / NORMAL TEXT
========================================== */

.stMarkdown p {
    color: #34495e !important;
    font-size: 16px;
}


/* ==========================================
   FILE UPLOADER
========================================== */

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 2px dashed #668bb5 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 15px rgba(30, 70, 110, 0.08);
}


/* File uploader text */

[data-testid="stFileUploader"] * {
    color: #263b50 !important;
}


/* ==========================================
   TEXT INPUT
========================================== */

[data-testid="stTextInput"] {
    margin-top: 10px;
}


[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #1f2937 !important;
    border: 2px solid #a8bdd3 !important;
    border-radius: 12px !important;
    padding: 13px !important;
    font-size: 16px !important;
}


/* Input focus */

[data-testid="stTextInput"] input:focus {
    border-color: #477bb5 !important;
    box-shadow: 0 0 0 3px rgba(71, 123, 181, 0.15) !important;
}


/* Input label */

[data-testid="stTextInput"] label {
    color: #173b67 !important;
    font-weight: 650 !important;
}


/* ==========================================
   BUTTONS
========================================== */

.stButton > button {
    background: linear-gradient(
        135deg,
        #386fa4,
        #28527a
    ) !important;

    color: white !important;

    border: none !important;
    border-radius: 10px !important;

    padding: 10px 25px !important;

    font-size: 16px !important;
    font-weight: 600 !important;

    transition: 0.2s ease;
}


.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #28527a,
        #1d3d5c
    ) !important;

    transform: translateY(-1px);
}


/* ==========================================
   SUCCESS / INFO BOXES
========================================== */

[data-testid="stAlert"] {
    border-radius: 12px !important;
}


/* ==========================================
   EXPANDER
========================================== */

[data-testid="stExpander"] {
    background-color: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid #c6d5e4 !important;
    border-radius: 12px !important;
}


/* Expander text */

[data-testid="stExpander"] * {
    color: #263b50 !important;
}


/* ==========================================
   HEADINGS
========================================== */

h2, h3 {
    color: #173b67 !important;
}


/* ==========================================
   DIVIDER
========================================== */

hr {
    border-color: #c8d6e4 !important;
}


/* ==========================================
   DARK MODE
========================================== */

@media (prefers-color-scheme: dark) {

    .stApp {
        background: linear-gradient(
            135deg,
            #101820 0%,
            #17232f 50%,
            #101820 100%
        );
    }

    h1 {
        color: #8fc7ff !important;
    }

    h2, h3 {
        color: #8fc7ff !important;
    }

    .stMarkdown p {
        color: #d6e2ee !important;
    }

    [data-testid="stFileUploader"] {
        background: #1c2936 !important;
        border-color: #6d9dcc !important;
    }

    [data-testid="stFileUploader"] * {
        color: #e2edf7 !important;
    }

    [data-testid="stTextInput"] input {
        background-color: #1c2936 !important;
        color: #f1f5f9 !important;
        border-color: #587795 !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #82b9e8 !important;
    }

    [data-testid="stTextInput"] label {
        color: #9bc9f5 !important;
    }

    [data-testid="stExpander"] {
        background-color: #1c2936 !important;
        border-color: #42586d !important;
    }

    [data-testid="stExpander"] * {
        color: #dce8f3 !important;
    }

    hr {
        border-color: #3b5063 !important;
    }

}

</style>
""", unsafe_allow_html=True)
# ==========================================
# TITLE
# ==========================================

st.title("📄 Chat with Your PDF")

st.write(
    "Ask questions about your PDF using RAG + Ollama"
)


# ==========================================
# TEST MESSAGE
# ==========================================

st.info("Application is running successfully!")


# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    # Read PDF

    reader = PdfReader(uploaded_file)

    st.write(
        "Number of pages:",
        len(reader.pages)
    )

    # Extract text

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    # Check text

    if not text.strip():

        st.error(
            "No text could be extracted from this PDF."
        )

        st.stop()


    st.write(
        "Total characters:",
        len(text)
    )


    # ==========================================
    # CREATE CHUNKS
    # ==========================================

    chunk_size = 500

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ]

        if chunk.strip():

            chunks.append(chunk)


    st.write(
        "Number of chunks:",
        len(chunks)
    )


    # ==========================================
    # LOAD EMBEDDING MODEL
    # ==========================================

    with st.spinner(
        "Loading embedding model..."
    ):

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


    st.success(
        "Embedding model loaded!"
    )


    # ==========================================
    # CREATE EMBEDDINGS
    # ==========================================

    with st.spinner(
        "Creating embeddings..."
    ):

        embeddings = embedding_model.encode(
            chunks
        )


    st.success(
        "Embeddings created!"
    )


    # ==========================================
    # CREATE CHROMADB
    # ==========================================

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documents"
    )


    # ==========================================
    # STORE DOCUMENT
    # ==========================================

    ids = []

    for i in range(len(chunks)):

        ids.append(
            "chunk_" + str(i)
        )


    # Remove old data

    try:

        old_data = collection.get()

        if old_data["ids"]:

            collection.delete(
                ids=old_data["ids"]
            )

    except Exception:

        pass


    collection.add(

        ids=ids,

        documents=chunks,

        embeddings=embeddings.tolist()

    )


    st.success(
        "PDF stored successfully in ChromaDB!"
    )


    st.divider()


    # ==========================================
    # QUESTION
    # ==========================================

    question = st.text_input(
        "Ask a question about the PDF"
    )


    # ==========================================
    # ANSWER
    # ==========================================

    if question:

        with st.spinner(
            "Searching document..."
        ):

            # Create question embedding

            question_embedding = (
                embedding_model.encode(
                    [question]
                )
            )


            # Search ChromaDB

            results = collection.query(

    query_embeddings=
    question_embedding.tolist(),

    n_results=len(chunks)

)


            # Get documents

            relevant_chunks = (
                results["documents"][0]
            )


            # Combine context

            context = "\n\n".join(
                relevant_chunks
            )


        # ==========================================
        # SHOW RETRIEVED DATA
        # ==========================================

        with st.expander(
            "View Retrieved Context"
        ):

            st.write(context)


        # ==========================================
        # CREATE PROMPT
        # ==========================================

        prompt = f"""

You are a helpful document assistant.

Answer the user's question using the information
available in the document context.

If the user asks for a summary, overview, or brief
description, summarize the important information
from the document context.

If the answer cannot be found in the document,
say:

"I could not find the answer in the document."

Do not make up information.

Document Context:

{context}

User Question:

{question}

Answer:

"""

        # ==========================================
        # OLLAMA
        # ==========================================

        with st.spinner(
            "Ollama is generating the answer..."
        ):

            response = ollama.chat(

                model="llama3.2",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            )


        # ==========================================
        # DISPLAY ANSWER
        # ==========================================

        answer = response["message"]["content"]

        st.subheader("🤖 Answer")

        st.write(answer)
