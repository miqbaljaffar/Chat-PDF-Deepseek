# Chat with PDF
A simple RAG (Retrieval-Augmented Generation) system using Deepseek, LangChain, and Streamlit to chat with PDFs and answer complex questions about your local documents.

Using the great tutorial linked here : [YouTube](https://youtu.be/M6vZ6b75p9k).

# Pre-requisites
Install Ollama on your local machine from the [official website](https://ollama.com/).

Pull the Deepseek model:

```bash
ollama pull deepseek-r1:1.5b
```

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

# Run
Run the Streamlit app:

```bash
streamlit run pdf_rag.py
```


# Steps

- First, we need to upload our PDFs so the user will come and upload their PDFs in our system (using StreamLit). This will be done in the upload_pdf function.
- Then, we need to load the PDFs and extract text from them. This is when we are going to use PDFPlumber and LangChain loaders to do it for us.
- In the next step, we are going to split the text into smaller chunks.
- After this preprocessing, we will need to index our documents, which means to bring them into vector spaces and store it inside the vector store.
- Then, we will implement a chat where a user can ask questions. We'll need to load all the documents related to that query in the vector store.
- At the end, we will create a prompt that will contain the user question and all the related documents that we retrieved, and we will ask the LLM to answer the user's question with the given context.
