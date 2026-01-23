import streamlit as st
import json
import os
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document

# Internal Modules
from agents.workflow import AgentWorkflow
from retriever.retrieval import RetrieverBuilder
from config.settings import settings

# env variables
load_dotenv()

st.set_page_config(page_title="Lecture RAG Assistant", layout="wide")


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
    retriever = retriever_builder.build_hybrid_retriever(docs) # if bm25 fails, falls back to vector only

    workflow_agent = AgentWorkflow()
    
    return retriever, workflow_agent

# Streamlit Interface 

st.title("🎓 Lecture Assistant RAG")
st.markdown("Ask questions about the lecture. I will verify answers and show you the relevant slide.")

# Check for API Key
if not os.getenv("GEMINI_API_KEY"):
    st.error("⚠️ GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Sidebar: File Status
with st.sidebar:
    st.header("System Status")
    json_file = "lecture.json"
    
    if os.path.exists(json_file):
        st.success(f"✅ Data found: {json_file}")
    else:
        st.error(f"❌ {json_file} not found. Please run pre_process.py first.")
        st.stop()
        
    st.info("Loading system... (this may take a moment)")
    retriever, workflow_agent = initialize_system(json_file)
    if retriever:
        st.success("System Ready!")
    else:
        st.error("Failed to initialize system.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], caption=f"Slide at {message['timestamp']}s", width=400)
        # Show debug info if available in history
        if "debug_docs" in message:
            with st.expander("Retrieved Context (Debug)"):
                for i, doc in enumerate(message["debug_docs"]):
                    st.markdown(f"**Chunk {i+1} (Time: {doc.metadata.get('start')}s):**")
                    st.text(doc.page_content)


if prompt := st.chat_input("What did the lecturer say about...?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*Thinking... (Researching & Verifying)*")
        
        try:
            workflow_graph = workflow_agent.create_workflow()
            
            debug_docs = retriever.invoke(prompt) 
            
            initial_state = {
                "question": prompt,
                "documents": debug_docs, # Pass retrieved docs directly to state
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
            
            # Show debug expander
            with st.expander("Retrieved Context (Debug)"):
                if not debug_docs:
                    st.warning("No relevant documents were retrieved! The database might be empty or the query is too distinct.")
                for i, doc in enumerate(debug_docs):
                    st.markdown(f"**Chunk {i+1} (Time: {doc.metadata.get('start')}s):**")
                    st.caption(doc.page_content[:300] + "...") # Preview
            
            history_entry = {
                "role": "assistant", 
                "content": answer,
                "debug_docs": debug_docs # Save for history view
            }
            if top_image:
                history_entry["image"] = top_image
                history_entry["timestamp"] = timestamp
            
            st.session_state.messages.append(history_entry)

        except Exception as e:
            message_placeholder.error(f"An error occurred: {str(e)}")