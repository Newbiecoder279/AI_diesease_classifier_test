#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# In[ ]:


load_dotenv()  # loads from .env file if you have one
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "IzaSyD7vUx5Fa9QUt56aLWh2uerbAC3Ad7UEi4")


# In[14]:


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def setup_qa_system(file_path: str):
    # Load PDF
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load_and_split()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Create embeddings (Gemini embeddings)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Build Chroma vectorstore
    vector_store = Chroma.from_documents(
        chunks, embeddings, persist_directory="./chroma_db"
    )

    retriever = vector_store.as_retriever()

    # ✅ Use Gemini as LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Conversational RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )

    return qa_chain




# In[15]:


if __name__ == "__main__":
    # Change file name to your PDF
    qa_chain = setup_qa_system(file_path="disease_treatments.pdf")

    print("💬 Ask questions about the PDF (type 'exit' to quit)\n")

    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break

        answer = qa_chain.run(question)
        print("\nAnswer:", answer, "\n")

