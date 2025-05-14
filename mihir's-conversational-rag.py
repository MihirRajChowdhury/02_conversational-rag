import os 
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

google_api_key = os.environ["GOOGLE_API_KEY"]
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq


# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=google_api_key)
llm = ChatGroq(model_name="llama-3.1-8b-instant")
from langchain.document_loaders import TextLoader


loader = TextLoader("./data/be-good.txt")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
split = text_splitter.split_documents(docs)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
vector_store = Chroma.from_documents(documents=split, embedding=embedding)
retriever = vector_store.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt) # this chain sends your prompts to your llm
rag_chain = create_retrieval_chain(retriever,question_answer_chain)# this is a rag chain which combines with a qa_chain able to ask questions to the retriever and then format the response with the llm

output = rag_chain.invoke({"input":"What is this article about"})

print("================================output======================")
# print(output["answer"])

output = rag_chain.invoke({"input":"What was the previous question about"})
# print(output["answer"])

from langchain_core.prompts import MessagesPlaceholder

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Given a chat history and the latest user question which might reference context in the chat history, "
     "formulate a standalone question which can be understood without the chat history. "
     "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

from langchain.chains import create_history_aware_retriever

history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

from langchain_core.messages import AIMessage, HumanMessage

# chat_history = []

# question = "What is this article about"

# ai_msg_1 = rag_chain.invoke({"input":question,"chat_history":chat_history})

# chat_history.extend([
#     HumanMessage(content=question),
#     AIMessage(content=ai_msg_1["answer"])
# ])

# second_question = "What was my first question about"

# ai_msg_2 = rag_chain.invoke({"input":second_question,"chat_history":chat_history})

# print(ai_msg_2["answer"])


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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

conversational_rag_chain.invoke(
    {"input": "What is this article about?"},
    config={
        "configurable": {"session_id": "001"}
    },  # constructs a key "001" in `store`.
)["answer"]

conversational_rag_chain.invoke(
    {"input": "What was my previous question about?"},
    config={"configurable": {"session_id": "001"}},
)["answer"]

for message in store["001"].messages:
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "User"

    print(f"{prefix}: {message.content}\n")