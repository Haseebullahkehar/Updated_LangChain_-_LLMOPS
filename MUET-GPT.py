import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Load and process document
loader = PyPDFLoader("data/Muet_Prospectus-23.pdf")
loaded_document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=5)
chunks = text_splitter.split_documents(loaded_document)

# Generate embeddings
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 1})

# Define prompt template
template = """Answer appropriately for the following questions based on the following context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Real-time search tool
search = TavilySearchResults(max_results=1)
tools = [search]
llm = ChatOpenAI(model="gpt-3.5-turbo")
agent_executor = create_react_agent(llm, tools)

# Streamlit UI
st.set_page_config(page_title="MUET Chatbot", page_icon="📘")

# Add MUET logo
st.image("data/Muet-logo.png", width=150)

st.title("MUET Chatbot")
st.write("Ask a question about Mehran University of Engineering and Technology")

user_input = st.text_input("Enter your query:")

if user_input:
    response = agent_executor.invoke(
        {"messages": [HumanMessage(content=user_input)]})
    if "messages" in response and isinstance(response["messages"], list) and len(response["messages"]) > 0:
        # Extracting final AI message content
        answer = response["messages"][-1].content
    else:
        answer = "No response received."
    st.write("**Answer:**", answer)
