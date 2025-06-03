from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize Ollama LLM
# Specify the model you pulled. For Phi-3 Mini (3.8B), it's usually 'phi3:mini'.
# Ollama will automatically try to use your GPU if available and compatible.
llm = OllamaLLM(model="phi3:mini")

# 2. Define a simple prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "{input}")
])

# 3. Create a simple chain
chain = prompt | llm | StrOutputParser()

# 4. Invoke the chain
if __name__ == "__main__":
    print("Welcome to Basic Phi-3 Chat! Type 'exit' to quit.")
    # You might want to monitor your GPU usage (e.g., Task Manager on Windows, nvidia-smi on Linux)
    # after you type your first input.
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print("Phi-3 is thinking...")
        response = chain.invoke({"input": user_input})
        print(f"Phi-3: {response}")