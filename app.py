
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask questions about Zyro Dynamics HR policies.")

PDF_PATH = "zyro-dynamics-hr-corpus"

@st.cache_resource
def load_rag():

    loader = PyPDFDirectoryLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":4}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the context.

Context:
{context}

Question:
{question}
""")

    return retriever,llm,prompt

retriever,llm,prompt = load_rag()

question = st.chat_input("Ask your HR question...")

if question:

    docs = retriever.invoke(question)

    context = "".join(
        d.page_content for d in docs
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "context":context,
        "question":question
    })

    st.chat_message("user").write(question)

    st.chat_message("assistant").write(answer)

    st.subheader("Sources")

    for d in docs:
        st.write(
            os.path.basename(
                d.metadata["source"]
            )
        )
