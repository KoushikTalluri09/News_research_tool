import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()
file_path = "faiss_store_openai.pkl"
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placefolder= st.empty()

llm=OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    Loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading Started ...")
    data = Loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size= 1000
    )
    main_placefolder.text("Data Splitting Started ...")
    docs= text_splitter.split_documents(data)

    embedings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})
    vectorstore_openai= FAISS.from_documents(docs,embedings)
    main_placefolder.text("Data Embeding Started ...")
    time.sleep(2)
    with open(file_path, "wb")as f :
        pickle.dump(vectorstore_openai,f)

query = main_placefolder.text_input("Question :")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources","")
            if sources:
                st.subheader("Sources")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
