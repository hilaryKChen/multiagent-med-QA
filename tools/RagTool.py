from crewai.tools import tool
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"  # Replace with your actual OpenAI API key

@tool("MY RAG TOOL")
def rag_tool(question: str) -> str:
    """Search medical database for relevant context"""
    embedding = OpenAIEmbeddings()
    db = Chroma(collection_name="vdb",persist_directory='./VectorStore2', embedding_function=embedding)
    #print(db._collection.count())
    results = db.similarity_search(question , k=3)  # Return the most similar 3 documents
    contents = []
    for doc in results:
        contents.append(doc.page_content)
    return "\n".join(contents)
