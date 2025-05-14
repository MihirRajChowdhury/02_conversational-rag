import os
from dotenv import load_dotenv, find_dotenv

# Ensure environment variables are loaded
_ = load_dotenv(find_dotenv())
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key exists
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in your .env file.")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

# Verify document path exists before loading
doc_path = "./data/be-good.txt"
if not os.path.exists(doc_path):
    raise FileNotFoundError(f"Document path not found: {doc_path}. Please check the path.")

# Load and split documents
loader = TextLoader(doc_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embeddings and Vector Store
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Use a persistent path for the Chroma database
vector_db_path = "./chroma_db"
os.makedirs(vector_db_path, exist_ok=True)

# Create or load the vector store
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embedding,
    persist_directory=vector_db_path
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 most relevant chunks

# Prompt for answer generation
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Chain for answering questions
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Prompt for reformulating questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Given a chat history and the latest user question which might reference context in the chat history, "
     "formulate a standalone question which can be understood without the chat history. "
     "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# History-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Full RAG chain with history
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Simulated in-memory chat history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- RAG Conversation Testing Function ---

def ask_question(question, session_id="001"):
    """Function to ask questions and display results consistently"""
    print(f"\n========== USER: {question} ==========\n")
    
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    
    print(f"ASSISTANT: {response['answer']}\n")
    return response

# Test with a few questions
ask_question("What is this article about?")
ask_question("What was my previous question about?")
ask_question("Can you tell me more specific details from the article?")

# Show the conversation history
print("\n========== CONVERSATION HISTORY ==========\n")
for message in store["001"].messages:
    prefix = "ASSISTANT" if isinstance(message, AIMessage) else "USER"
    print(f"{prefix}: {message.content}\n")