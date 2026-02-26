## rag Q&A conversation with pdf including Chat Histrory
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv
load_dotenv()


os.environ["HUGGINGFACEHUB_API_TOKEN"]=os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-miniLM-L6-v2")


## setup Streamlit
st.title("🧠Conversatinal Rag with PDF uplods and Chat history ")
st.write("uplode PDf's and chat with thear content")

## input the groq api key
api_key=st.text_input("Enter Groq API Key:",type="password")

## check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant")
    session_id=st.text_input("Enter Session ID for Chat History:",value="default_session")
    ##
    if 'store' not in st.session_state:
        st.session_state.store={}

    uplode_files=st.file_uploader("Choose A PDf File",type="pdf",accept_multiple_files=True)
    #process the uplode file
    if uplode_files:
        documents=[]
        for uplode_file in uplode_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uplode_file.getvalue())
                file_name=uplode_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
    #split and creat emabedding  for  documents
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        splits= text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)     
        retriever=vectorstore.as_retriever()  
    ## system prompt
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question"
            "just reformulate it if needed and otherwise return it as is." 
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        history_aware_retriever=create_history_aware_retriever(llm ,retriever,contextualize_q_prompt)
    ## Answare qustion prompt
    
        system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer, say that you"
            "don't know. Use three sentences maximum and keep the."
            "answer concise.\n\n"
            "{context}"
        )
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
            )
        qustion_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,qustion_answer_chain)
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("Your Question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable": {"session_id": session_id}

                },

            )
            st.write(st.session_state.store)
            st.success(f"Assistant: {response['answer']}")
            st.write("Chat History:",session_history.messages)
else:
    st.warning("Please enter your Groq API Key to continue.")