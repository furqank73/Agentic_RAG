import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_groq import ChatGroq
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langsmith import Client
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from IPython.display import Image, display

# ==================== Streamlit Page Setup ==================== #
st.set_page_config(page_title="Agentic RAG with LangGraph", layout="wide")
st.title("🤖 Agentic RAG System using LangGraph + Groq")
st.write("This app uses **LangGraph**, **LangChain**, and **Groq LLM** to retrieve and answer from LangChain + LangGraph documentation using an agentic workflow.")

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ==================== Sidebar ==================== #
st.sidebar.title("Settings")
rebuild = st.sidebar.checkbox("Rebuild Vector DB", value=False)
user_query = st.text_input("Enter your question:", "how to create a simple hello world example: on langgraph")

# ==================== Document Loading & Vectorization ==================== #
@st.cache_resource(show_spinner=True)
def build_vectorstores():
    st.info("🔍 Loading and indexing documents... This might take a few minutes on first run.")
    langgraph_urls = [
        "https://docs.langchain.com/oss/python/langgraph/overview",
        "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
        "https://docs.langchain.com/oss/python/langgraph/graph-api#map-reduce-and-the-send-api"
    ]
    langchain_urls = [
        "https://python.langchain.com/docs/tutorials/",
        "https://python.langchain.com/docs/tutorials/chatbot/",
        "https://python.langchain.com/docs/tutorials/qa_chat_history/"
    ]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # LangGraph Docs
    langgraph_docs = [WebBaseLoader(url).load() for url in langgraph_urls]
    lg_docs_list = [item for sublist in langgraph_docs for item in sublist]
    lg_doc_splits = text_splitter.split_documents(lg_docs_list)
    langgraph_vectorstore = FAISS.from_documents(lg_doc_splits, embedding=embeddings)
    langGraph_retriever = langgraph_vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(langGraph_retriever, "retriever_vector_langGraph", "Search and run information about LangGraph")

    # LangChain Docs
    langchain_docs = [WebBaseLoader(url).load() for url in langchain_urls]
    lc_docs_list = [item for sublist in langchain_docs for item in sublist]
    lc_doc_splits = text_splitter.split_documents(lc_docs_list)
    langchain_vectorstore = FAISS.from_documents(lc_doc_splits, embedding=embeddings)
    langchain_retriever = langchain_vectorstore.as_retriever()
    langchain_retriever_tool = create_retriever_tool(langchain_retriever, "retriever_vector_langchain", "Search and run information about LangChain")

    return retriever_tool, langchain_retriever_tool

retriever_tool, langchain_retriever_tool = build_vectorstores()

# ==================== LLM Setup ==================== #
llm = ChatGroq(model="llama-3.1-8b-instant")
tools = [retriever_tool, langchain_retriever_tool]

# ==================== Agentic Graph Setup ==================== #
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state):
    st.write("🧠 Running Agent...")
    messages = state["messages"]
    model = ChatGroq(model="llama-3.1-8b-instant").bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

def grade_documents(state) -> Literal["generate", "rewrite"]:
    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatGroq(model="llama-3.1-8b-instant")
    llm_with_tool = model.with_structured_output(grade)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        Here is the retrieved document:\n\n{context}\n\n
        Here is the user question: {question}\n
        If relevant, return 'yes', else 'no'.""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_tool
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    return "generate" if scored_result.binary_score == "yes" else "rewrite"

client = Client(api_key="your_langsmith_api")

def generate(state):
    st.write("💡 Generating final answer...")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
    llm = ChatGroq(model="llama-3.1-8b-instant")
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def rewrite(state):
    st.write("✍️ Rewriting question...")
    messages = state["messages"]
    question = messages[0].content
    msg = [HumanMessage(content=f"Improve this question for clarity:\n{question}")]
    model = ChatGroq(model="llama-3.1-8b-instant")
    response = model.invoke(msg)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool, langchain_retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")
graph = workflow.compile()

# ==================== Run Query ==================== #
if st.button("Run Agentic RAG"):
    if user_query.strip():
        with st.spinner("Running Agentic RAG workflow..."):
            result = graph.invoke({"messages": [HumanMessage(content=user_query)]})
            st.success("✅ Done!")
            st.subheader("Final Answer:")
            st.write(result["messages"][-1].content)
    else:
        st.warning("Please enter a question to query.")

st.caption("Built with ❤️ using LangChain, LangGraph, Groq, and Streamlit.")
