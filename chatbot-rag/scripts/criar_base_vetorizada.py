import os
import shutil
import stat
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Caminho do PDF e pasta onde vamos salvar os vetores
pdf_path = "data/pokemons_primeira_geracao.pdf"

def handle_remove_readonly(func, path, exc):
    # Remove arquivos somente leitura
    os.chmod(path, stat.S_IWRITE)
    func(path)

persist_directory = "db"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory, onerror=handle_remove_readonly)
    print(f"🧹 Base vetorizada antiga removida de '{persist_directory}'.")

# Carregar o PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Dividir o texto em pedaços menores
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# Criar embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Criar e salvar a base vetorizada com Chroma
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()

print("✅ Base vetorizada criada com sucesso e salva em 'db/'!")
