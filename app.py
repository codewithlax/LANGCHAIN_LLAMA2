#Because OLLAMA is a 3rd party LLM, you get it from langchain_community (vs let's say langchain_openai import ChatOpenAI)
from langchain_community.llms import Ollama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
# To enable all tracing features (Langsmith - for tracking)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
print(os.environ["LANGCHAIN_API_KEY"])

# Prompt Template
prompt = ChatPromptTemplate.from_messages(

    [
        ("system","""You are going to be a funny assistant. Please respond to the user queries with a joke"""),
        ("user","Question: {question}")

    ]
)

# Streamlit:
st.title("Langchain Demo with OLLAMA")
input_text = st.text_input("Shoot your question")


# LLM: mostly follows this syntax: llm = LLMLIBRARY(model="SPECIFIC LLM")
llm = Ollama(model="llama2") # You can check installation and working in cmd: ollama run llama2 "Hello, how are you?"  
output_parser = StrOutputParser()
chain = prompt|llm|output_parser 


if input_text:
    st.write(chain.invoke({'question':input_text}))

















# import os
# from langchain.vectorstores import FAISS
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import RetrievalQA
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Step 1: Load and Split Documents
# # Load a sample text file as the document
# loader = TextLoader("C:/Users/91828/Documents/LLMs/cpu_vs_gpu/sample.txt")  # Replace with your file path
# documents = loader.load()

# # Step 2: Split Documents into Chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100
# )
# texts = text_splitter.create_documents(documents[0].page_content)

# # Step 3: Create Embeddings
# # Use a free open-source model like distilbert-base-uncased for embeddings
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Free and open-source
# # Embed the documents
# # embedded_docs = embedding_model.embed_documents([doc.page_content for doc in texts])

# # Step 4: Initialize FAISS Vector Store
# db = FAISS.from_documents(texts, embedding_model)

# # Create a retriever for document search
# retriever = db.as_retriever()

# # Step 5: Set up the Language Model (LLM)
# # Use GPT-Neo, GPT-J, or Falcon from HuggingFace
# model_name = "EleutherAI/gpt-neo-1.3B"  # You can use "gpt-j-6B" or "tiiuae/falcon-7b-instruct" as well
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# llm_model = AutoModelForCausalLM.from_pretrained(model_name)

# # Step 6: Define a Prompt Template for Question Answering
# prompt = ChatPromptTemplate.from_template(
#     """
#     Based on the following context, answer the user's query.

#     Context: {context}
#     Question: {input}
#     Answer:
#     """
# )

# # Step 7: Create the RAG Chain (Retrieval + LLM)
# retrieval_chain = RetrievalQA.from_chain_type(
#     llm_model=llm_model,
#     retriever=retriever,
#     prompt_template=prompt,
#     tokenizer=tokenizer
# )

# # Step 8: Run a Query
# query = "What is the best programming language for data science?"
# result = retrieval_chain.run({"input": query})
# print(result)
