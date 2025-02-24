import streamlit as st

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

pdfs_directory = './pdfs/' # the user will upload a pdf that will be stored here

embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b") # create embeddings (vectors out of texts)
vector_store = InMemoryVectorStore(embeddings) # create a vector store (could be using Chroma, ElasticSearch...)

model = OllamaLLM(model="deepseek-r1:1.5b")

def upload_pdf(file): 
    ''' uploads the pdf given by the user '''
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())
        
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    
    return documents

def split_text(documents):
    ''' splits the document into smaller chunks'''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents) 

def index_docs(documents):
    ''' indexes the documents - uses the embeddings to move the docs into vector space'''
    vector_store.add_documents(documents)
    
def retrieve_docs(query):
    ''' retrieves all the documents related to the user's question using similarity search'''
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents]) # joining all the documents together to pass them to the LLM
    prompt = ChatPromptTemplate.from_template(template) # using a template from langchain (defined above)
    chain = prompt | model
    
    return chain.invoke({"question": question, "context":context})


uploaded_file = st.file_uploader(
    "Upload PDF", 
    type="pdf", 
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)
    
    question = st.chat_input()
    
    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)