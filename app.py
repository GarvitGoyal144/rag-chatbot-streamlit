import streamlit as st # <-- This should be the first import

# --- Streamlit UI (Page Configuration FIRST) ---
st.set_page_config(page_title="RAG Chatbot", layout="centered")

import os
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain_core.documents import Document
import json
import requests # For fetching website content robustly

# Load environment variables (for SerpApi key)
load_dotenv()
os.environ["USER_AGENT"] = "MyAIChatbot/1.0 (Contact: basic@gmail.com)" # Or something descriptive
os.environ['LOCATION'] = "Prayagraj, Uttar Pradesh, India" # Set default location for weather queries etc.

# --- Configuration ---
OLLAMA_MODEL = "phi3:mini"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
VECTOR_DB_DIR = "./chroma_db" # Directory to store your vector database
CHUNK_SIZE = 1000 # Size of text chunks
CHUNK_OVERLAP = 200 # Overlap between chunks

# --- 1. Initialize LLMs and Embeddings (using st.cache_resource) ---
@st.cache_resource
def initialize_llm_and_embeddings():
    st.write("Initializing LLM and Embeddings (this happens once)...")

    # --- START OF CHANGES ---
    # Get OLLAMA_HOST from environment variable (Streamlit Secrets will provide this)
    ollama_host_url = os.getenv("OLLAMA_HOST")
    if not ollama_host_url:
        st.error("OLLAMA_HOST environment variable not set. LLM and Embeddings will not work.")
        # Return None for llm and embeddings if initialization fails
        return None, None

    # Initialize Ollama LLM and Embeddings, passing the ngrok URL as base_url
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL, base_url=ollama_host_url, temperature=0.7)
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=ollama_host_url)
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing Ollama: {e}. Check OLLAMA_HOST and if Ollama server is running with the correct models.")
        return None, None
    # --- END OF CHANGES ---

llm, embeddings = initialize_llm_and_embeddings()

# --- NEW: Check if LLM/Embeddings initialized successfully before proceeding with UI
if llm is None or embeddings is None:
    st.stop() # Stop the app if initialization failed to prevent further errors

# --- 2. Set up the Web Search Tool (using SerpApi) ---
@st.cache_resource
def initialize_serpapi_wrapper():
    st.write("Initializing SerpAPI Wrapper (this happens once)...")
    # Ensure SERPAPI_API_KEY is available in environment variables
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        st.error("SERPAPI_API_KEY environment variable not set. Web search will not work.")
        return None
    search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    return Tool(
        name="Web Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or fresh information. Input should be a search query."
    )

web_search_tool = initialize_serpapi_wrapper()

# --- 3. RAG Pipeline Components ---

# This prompt is used for the LLM to answer questions given retrieved context.
document_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's questions based *only* on the provided context. If the answer is not in the context, state that you don't know."),
    ("user", "Context: {context}\n\nQuestion: {input}")
])

# Define a function to process a search query and load content
def retrieve_and_process_web_data(query: str):
    if not web_search_tool:
        st.warning("SerpAPI not initialized. Cannot perform web search.")
        return None
    # --- NEW: Also check embeddings here ---
    if not embeddings:
        st.error("Embedding model not initialized. Cannot create vector store.")
        return None
    # --- END NEW ---

    st.info(f"Performing web search for: '{query}'")
    try:
        raw_search_results = web_search_tool.run(query)
    except Exception as e:
        st.error(f"Error during SerpAPI search: {e}. Check your API key and network.")
        return None

    urls_to_load = []
    text_from_direct_answer = None

    search_data = raw_search_results

    if isinstance(search_data, dict):
        if search_data.get("organic_results"):
            for i, result in enumerate(search_data["organic_results"][:3]):
                if result.get("link"):
                    urls_to_load.append(result["link"])
        elif search_data.get("answer_box") and search_data["answer_box"].get("link"):
            urls_to_load.append(search_data["answer_box"]["link"])
        elif search_data.get("knowledge_graph") and search_data["knowledge_graph"].get("url"):
            urls_to_load.append(search_data["knowledge_graph"]["url"])
        elif search_data.get("answer_box") and search_data["answer_box"].get("answer"):
            text_from_direct_answer = search_data["answer_box"]["answer"]
        elif search_data.get("answer_box") and search_data["answer_box"].get("snippet"):
            text_from_direct_answer = search_data["answer_box"]["snippet"]
        elif search_data.get("knowledge_graph") and search_data["knowledge_graph"].get("description"):
            text_from_direct_answer = search_data["knowledge_graph"]["description"]
        else:
            st.write("No structured results (organic, answer box, knowledge graph) found.")
    else:
        text_from_direct_answer = str(search_data)
        st.write(f"SerpAPI returned direct text answer: '{text_from_direct_answer[:100]}...'")

    all_docs = []

    if text_from_direct_answer:
        all_docs.append(Document(page_content=text_from_direct_answer, metadata={"source": "serpapi_direct_answer"}))
        st.info("Using direct text answer from SerpAPI.")
    elif not urls_to_load:
        st.warning("No relevant URLs found in search results or direct answers.")
        return None

    if urls_to_load:
        st.info(f"Found {len(urls_to_load)} URLs. Loading content...")
        for url in urls_to_load:
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    if docs:
                        all_docs.extend(docs)
                    else:
                        st.warning(f"No content loaded from {url}")
                else:
                    st.warning(
                        f"Skipping {url}: Not HTML or inaccessible (Status: {response.status_code}, Content-Type: {response.headers.get('Content-Type')})"
                    )
            except requests.exceptions.RequestException as e:
                st.error(f"Error accessing {url}: {e}")
            except Exception as e:
                st.error(f"Error loading content from {url}: {e}")

    if not all_docs:
        st.warning("No content successfully loaded from any URL or no direct answer was available.")
        return None

    st.success(f"Loaded {len(all_docs)} document(s) from the web.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_docs = text_splitter.split_documents(all_docs)
    st.info(f"Split into {len(chunked_docs)} chunks.")

    vectorstore = Chroma.from_documents(chunked_docs, embeddings)
    st.info("Vector store created.")
    return vectorstore.as_retriever()

# --- 4. Define the overall conversational flow ---
qa_chain = create_stuff_documents_chain(llm, document_qa_prompt)

def get_answer_with_web_context(user_question: str):
    keywords = ['what is', 'who is', 'when did', 'how many', 'latest', 'current', 'news', 'update', 'facts about',
                'today', 'tomorrow', 'live', 'score', 'match', 'ipl', 'nba', 'football', 'cricket', 'weather']
    needs_web_search = any(kw in user_question.lower() for kw in keywords) or '?' in user_question

    if needs_web_search:
        st.info(f"Query '{user_question}' seems to require web search.")
        search_query = user_question

        if 'ipl match today' in user_question.lower() or 'ipl today' in user_question.lower():
            search_query = 'IPL match today teams schedule'
        elif 'ipl' in user_question.lower() and ('match' in user_question.lower() or 'teams' in user_question.lower()):
            search_query = user_question + ' schedule'
        elif 'weather today' in user_question.lower():
            search_query = f"weather in {os.getenv('LOCATION', 'Prayagraj, India')} today"

        retriever = retrieve_and_process_web_data(search_query)
        if retriever:
            retrieval_chain = create_retrieval_chain(retriever, qa_chain)
            try:
                response = retrieval_chain.invoke({"input": user_question})
                return response.get("answer", "No answer found based on web search.")
            except Exception as e:
                st.error(f"Error during retrieval chain invocation: {e}")
                return "An error occurred while processing web-based retrieval."
        else:
            return "Could not retrieve relevant web information for your query."
    else:
        st.info("Query does not seem to require web search. Answering with general knowledge.")
        direct_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("user", "{input}")
        ])
        direct_chain = direct_prompt | llm | StrOutputParser()
        return direct_chain.invoke({"input": user_question}) # Changed from user_input to user_question


# --- Streamlit UI ---

st.title("🤖 RAG Chatbot with Web Search")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an initial greeting from the assistant
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm a RAG chatbot. Ask me anything, especially about current events or things requiring a web search!"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What do you want to know?"):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."): # Show a spinner while AI processes
            ai_response = get_answer_with_web_context(prompt)
            st.markdown(ai_response)
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})