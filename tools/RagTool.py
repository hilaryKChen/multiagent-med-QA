from crewai.tools import tool
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-iY1j43dZqk8b9ReDULM6zzYN3GJ85eHLfFUi6srBUaEZDRJm"

@tool("MY RAG TOOL")
def rag_tool(question: str) -> str:
    """Search medical database for relevant context"""
    embedding = OpenAIEmbeddings()
    db = Chroma(collection_name="vdb",persist_directory='./VectorStore2', embedding_function=embedding)
    #print(db._collection.count())
    results = db.similarity_search(question , k=3)  # 返回最相似的 2 个文档
    contents = []
    for doc in results:
        contents.append(doc.page_content)
    return "\n".join(contents)
