from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import wikipedia


os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-iY1j43dZqk8b9ReDULM6zzYN3GJ85eHLfFUi6srBUaEZDRJm"

# Sample abstracts from PubMed
pubmed_dataset = load_dataset("ccdv/pubmed-summarization", split="train[:100000]")
docs = [d['abstract'] for d in pubmed_dataset if 'abstract' in d]

wiki_topics = ['Anesthesia', 'Anatomy', 'Biochemistry', 'Dentistry', 'Otorhinolaryngology', 'Forensic_medicine', 
               'Obstetrics_and_gynaecology', 'Medicine', 'Microbiology','Ophthalmology', 'Orthopedic_surgery', 
               'Pathology', 'Pediatrics', 'Pharmacology', 'Physiology', 'Psychiatry','Radiology', 'Surgery', 
               'Skin_condition', 'Preventive_and_social_medicine']

wiki_docs = []

for topic in wiki_topics:
    try:
        content = wikipedia.page(topic).content
        wiki_docs.append(Document(page_content=content, metadata={'source': 'Wikipedia', 'title': topic}))
    except Exception as e:
        print(f"Error fetching {topic}: {e}")

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=5)
pubmed_split = text_splitter.split_documents([Document(page_content=doc) for doc in docs])
wiki_split = text_splitter.split_documents(wiki_docs)
split_docs = pubmed_split + wiki_split
# Initialize OpenAI embeddings
embedding = OpenAIEmbeddings()
# Store in vector database
db = Chroma(collection_name="vdb", persist_directory='./VectorStore2', embedding_function=embedding)

# 分批添加文档，避免超出 Chroma 限制
def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

for chunk in batched(split_docs, 5000):
    db.add_documents(chunk)

# db.add_documents(split_docs)

# # Save the vector store to disk
# db.persist()
# # Load the vector store from disk
# db = Chroma(collection_name="vdb", persist_directory='./VectorStore', embedding_function=embedding)
# # Perform a similarity search
# results = db.similarity_search("What is the mechanism of action of aspirin?", k=3)  # Return the most similar 3 documents
# # Print the results
# for doc in results:
#     print(doc.page_content)
# print(db._collection.count())