import streamlit as st
import time
from langchain_core.messages import AIMessage,HumanMessage 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv
load_dotenv()


def stream_data(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.02)

def get_vectorstore_from_url(website_url):
  loader=WebBaseLoader(website_url)
  document=loader.load()  
  splitter=RecursiveCharacterTextSplitter()
  document_chunks=splitter.split_documents(document)
  vectorstore=FAISS.from_documents(document_chunks,OpenAIEmbeddings())
  return vectorstore 

def get_context_retriever_chain(vectorstore):
  llm=ChatOpenAI()
  retriever=vectorstore.as_retriever()

  prompt=ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up information relevant to the conversation"),
  ]
  )  

  retriever_chain=create_history_aware_retriever(llm,retriever, prompt)

  return retriever_chain 
  
def get_conversational_rag_chain(retriever_chain):
  llm=ChatOpenAI()
   
  prompt=ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the below context: \n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("user", "{input}"),
  ]
  )   
  stuff_documents_chain=create_stuff_documents_chain(llm,prompt)

  return create_retrieval_chain(retriever_chain,stuff_documents_chain) 

def get_response(user_query):
   
  retriever_chain=get_context_retriever_chain(st.session_state.vector_store)

  conversational_rag_chain=get_conversational_rag_chain(retriever_chain) 

  response = conversational_rag_chain.invoke({"chat_history": st.session_state.chat_history,
    "input": user_query}  ) 

  return response["answer"] 
 

st.set_page_config(page_title="Chat with Web", page_icon="🤖")
st.title("💬 Chat with Websites")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter website URL")

if website_url is None or website_url == "":
  st.info("Please enter a website URL")

else:
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
  if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url(website_url)   
  

  user_query=st.chat_input("Type a message...")

  if user_query is not None and user_query != "":
    response=get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query)) 
    st.session_state.chat_history.append(AIMessage(content=response))
     
    

  for i, message in enumerate(st.session_state.chat_history):
    is_last_message = (i == len(st.session_state.chat_history) - 1)

    if isinstance(message, AIMessage):
      with st.chat_message("AI"):
        if is_last_message: 
          st.write_stream(stream_data(message.content))
        else:
          st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

    
  