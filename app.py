import streamlit as st
import json
import os

from dotenv import load_dotenv 
load_dotenv()

from typing import List
from langchain.schema import Document
from agents.workflow import AgentWorkflow
from retriever.retrieval import RetrieverBuilder
from config.settings import settings

# --- Page Config ---
st.set_page_config(page_title="Lecture RAG Assistant", layout="wide")

# --- Helper Fns ---

@st.cache_resource
def load_and_process_data(json_path: str) -> List[Document]:
    """
    Loads lecture.json and converts it into LangChain Documents.
    Combines transcript and slide text for better retrieval context.
    """
    if not os.path.exists(json_path):
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for chunk in data:
        content = f"Transcript: {chunk['transcript']}\nSlide Content: {chunk['slide_text']}"
        
        metadata = {
            "start": chunk['start'],
            "end": chunk['end'],
            "slide_image": chunk['slide_image']
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents

@st.cache_resource
def initialize_system(json_path: str):
    """
    Initializes the Vector DB (Chroma) and the Agent Workflow.
    Cached to prevent reloading on every interaction.
    """
    docs = load_and_process_data(json_path)
    if not docs:
        return None, None

    retriever_builder = RetrieverBuilder()
    retriever = retriever_builder.build_hybrid_retriever(docs)

    workflow_agent = AgentWorkflow()
    
    return retriever, workflow_agent

# --- Main App Interface ---

st.title("🎓 Lecture Assistant RAG")
st.markdown("Ask questions about the lecture. I will verify answers and show you the relevant slide.")

# Check for API Key
if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found in environment variables. Please set it to continue.")
    st.stop()

# Sidebar: File Status
with st.sidebar:
    st.header("System Status")
    json_file = "lecture.json"
    
    if os.path.exists(json_file):
        st.success(f"Success! Data found: {json_file}")
    else:
        st.error(f"{json_file} not found. Please run pre_process.py first.")
        st.stop()
        
    st.info("Loading system...")
    retriever, workflow_agent = initialize_system(json_file)
    st.success("System Ready!")

#chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], caption=f"Slide at {message['timestamp']}s", width=400)

if prompt := st.chat_input("What did the lecturer say about...?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*Thinking... (Researching & Verifying)*")
        
        try:
            #using compiled graph directly to get the state, not just the text
            workflow_graph = workflow_agent.create_workflow()
            
            initial_state = {
                "question": prompt,
                "documents": [],  
                "draft_answer": "",
                "verification_report": "",
                "is_relevant": False,
                "retriever": retriever
            }
            
            final_state = workflow_graph.invoke(initial_state)
            
            answer = final_state.get("draft_answer", "Sorry, I couldn't generate an answer.")
            
            relevant_docs = final_state.get("documents", [])
            top_image = None
            timestamp = None
            
            if relevant_docs:
                top_doc = relevant_docs[0]
                top_image = top_doc.metadata.get("slide_image")
                timestamp = top_doc.metadata.get("start")

 
            message_placeholder.markdown(answer)
            
      
            if top_image and os.path.exists(top_image):
                st.image(top_image, caption=f"Reference Slide (Time: {timestamp}s)", width=500)
            
            history_entry = {"role": "assistant", "content": answer}
            if top_image:
                history_entry["image"] = top_image
                history_entry["timestamp"] = timestamp
            
            st.session_state.messages.append(history_entry)

        except Exception as e:
            message_placeholder.error(f"An error occurred: {str(e)}")