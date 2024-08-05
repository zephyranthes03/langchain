from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st

st.title("Langchain Ollama llama3.1 test")
template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

# Req command : ollama pull llama3.1 
model = OllamaLLM(model="llama3.1")
chain = prompt | model
question = st.chat_input("Enter your question here")
if question:
    st.write(chain.invoke({"question": question}))    
