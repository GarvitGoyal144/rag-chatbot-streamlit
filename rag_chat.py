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


# --- Configuration ---
OLLAMA_MODEL = "phi3:mini"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
VECTOR_DB_DIR = "./chroma_db" # Directory to store your vector database
CHUNK_SIZE = 1000 # Size of text chunks
CHUNK_OVERLAP = 200 # Overlap between chunks

# --- 1. Initialize LLMs and Embeddings ---
llm = OllamaLLM(model=OLLAMA_MODEL)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

# --- 2. Set up the Web Search Tool (using SerpApi) ---
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

# Create a LangChain Tool from the SerpAPIWrapper
web_search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="useful for when you need to answer questions about current events or fresh information. Input should be a search query."
)

# --- 3. RAG Pipeline Components ---

# This prompt is used for the LLM to answer questions given retrieved context.
document_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer the user's questions based *only* on the provided context. If the answer is not in the context, state that you don't know."),
    ("user", "Context: {context}\n\nQuestion: {input}")
])

# Define a function to process a search query and load content
def retrieve_and_process_web_data(query: str):
    print(f"\n--- Performing web search for: '{query}' ---")
    # web_search_tool is a global object from your original script
    # It calls SerpAPIWrapper.run(), which returns a dict directly if successful
    raw_search_results = web_search_tool.run(query)

    urls_to_load = []
    text_from_direct_answer = None

    # --- CHANGE START ---
    # SerpAPIWrapper's .run() method already parses the JSON into a dictionary.
    # No need for json.loads() here, which caused the TypeError.
    search_data = raw_search_results

    # Now, check if search_data is indeed a dictionary (which it should be from SerpAPIWrapper)
    # or if it's a direct string result (less common, but handled for robustness).
    if isinstance(search_data, dict):
        if search_data.get("organic_results"):
            # Get up to 3 relevant URLs
            for i, result in enumerate(search_data["organic_results"][:3]):
                if result.get("link"):
                    urls_to_load.append(result["link"])
        elif search_data.get("answer_box") and search_data["answer_box"].get("link"):
            urls_to_load.append(search_data["answer_box"]["link"]) # Try answer box link
        elif search_data.get("knowledge_graph") and search_data["knowledge_graph"].get("url"):
            urls_to_load.append(search_data["knowledge_graph"]["url"]) # Try knowledge graph
        elif search_data.get("answer_box") and search_data["answer_box"].get("answer"):
            text_from_direct_answer = search_data["answer_box"]["answer"] # Direct text answer
        elif search_data.get("answer_box") and search_data["answer_box"].get("snippet"):
            text_from_direct_answer = search_data["answer_box"]["snippet"] # Direct snippet
        elif search_data.get("knowledge_graph") and search_data["knowledge_graph"].get("description"):
            text_from_direct_answer = search_data["knowledge_graph"]["description"] # Knowledge graph description
        else:
            print("No structured results (organic, answer box, knowledge graph) found.")
    else:
        # If it's not a dictionary, assume SerpAPI returned a direct text answer as a string
        text_from_direct_answer = str(search_data) # Ensure it's a string
        print(f"SerpAPI returned direct text answer: '{text_from_direct_answer[:100]}...'")
    # --- CHANGE END ---

    all_docs = []

    if text_from_direct_answer:
        # Create a document from the direct text answer
        all_docs.append(Document(page_content=text_from_direct_answer, metadata={"source": "serpapi_direct_answer"}))
        print("--- Using direct text answer from SerpAPI. ---")
    elif not urls_to_load:
        print("No relevant URLs found in search results or direct answers.")
        return []

    if urls_to_load:
        print(f"--- Found {len(urls_to_load)} URLs. Loading content... ---")
        for url in urls_to_load:
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    if docs:
                        all_docs.extend(docs)
                    else:
                        print(f"No content loaded from {url}")
                else:
                    print(
                        f"Skipping {url}: Not HTML or inaccessible (Status: {response.status_code}, Content-Type: {response.headers.get('Content-Type')})"
                    )
            except requests.exceptions.RequestException as e:
                print(f"Error accessing {url}: {e}")
            except Exception as e:
                print(f"Error loading content from {url}: {e}")

    if not all_docs:
        print("No content successfully loaded from any URL or no direct answer was available.")
        return []

    print(f"--- Loaded {len(all_docs)} document(s) from the web. ---")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_docs = text_splitter.split_documents(all_docs)
    print(f"--- Split into {len(chunked_docs)} chunks. ---")

    # Create a vector store from the chunks
    vectorstore = Chroma.from_documents(chunked_docs, embeddings)
    print("--- Vector store created. ---")
    return vectorstore.as_retriever() # Return as a retriever object

# --- 4. Define the overall conversational flow ---
qa_chain = create_stuff_documents_chain(llm, document_qa_prompt)

def get_answer_with_web_context(user_question: str):
    keywords = ['what is', 'who is', 'when did', 'how many', 'latest', 'current', 'news', 'update', 'facts about',
                'today', 'tomorrow', 'live', 'score', 'match', 'ipl', 'nba', 'football', 'cricket', 'weather']
    needs_web_search = any(kw in user_question.lower() for kw in keywords) or '?' in user_question

    if needs_web_search:
        print(f"\n--- Query '{user_question}' seems to require web search. ---")
        # Formulate a better search query for current events/sports
        search_query = user_question # Default to user's question

        # Refine search query for specific cases
        if 'ipl match today' in user_question.lower() or 'ipl today' in user_question.lower():
            search_query = 'IPL match today teams schedule'
        elif 'ipl' in user_question.lower() and ('match' in user_question.lower() or 'teams' in user_question.lower()):
            search_query = user_question + ' schedule'
        # Add other sport/current event specific refinements as needed
        elif 'weather today' in user_question.lower():
            search_query = f"weather in {os.getenv('LOCATION', 'Prayagraj, India')} today" # Use environment variable for location

        retriever = retrieve_and_process_web_data(search_query)
        if retriever:
            retrieval_chain = create_retrieval_chain(retriever, qa_chain)
            response = retrieval_chain.invoke({"input": user_question})
            return response.get("answer", "No answer found based on web search.")
        else:
            return "Could not retrieve relevant web information for your query."
    else:
        print("\n--- Query does not seem to require web search. Answering with general knowledge. ---")
        direct_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("user", "{input}")
        ])
        direct_chain = direct_prompt | llm | StrOutputParser()
        return direct_chain.invoke({"input": user_input})


# --- 5. Main Loop ---
if __name__ == "__main__":
    print("Welcome to ChatGPT-like AI! Type 'exit' to quit.")
    print("Queries about current events will trigger a web search.")
    # Set default location for weather queries etc.
    os.environ['LOCATION'] = "Prayagraj, Uttar Pradesh, India"

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        response = get_answer_with_web_context(user_input)
        print(f"AI: {response}")

