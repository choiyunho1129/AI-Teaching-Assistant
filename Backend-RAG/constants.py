import os
# from dotenv import load_dotenv
from chromadb.config import Settings
# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain_community.document_loaders import (
    CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader,
    UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
)

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
TRANSCRIPT_DIRECTORY = f"{ROOT_DIRECTORY}/class_transcript"

# Default DB locations (aligned with Ingest_dualDB defaults)
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/answer_DB"
HINT_PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/hint_DB"
QUIZ_PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/quiz_DB"
ANSWER_COLLECTION_NAME = "answer_db"
HINT_COLLECTION_NAME = "hint_db"
QUIZ_COLLECTION_NAME = "quiz_db"
# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = 1024  # int(CONTEXT_WINDOW_SIZE/4)

#### If you get a "not enough space in the buffer" error, you should reduce the values below, start with half of the original values and keep halving the value until the error stops appearing
N_GPU_LAYERS = 100  # Llama-2-70B has 83 layers
N_BATCH = 512

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B" # Uses 0.2 GB of VRAM (Less accurate but fastest - only requires 150mb of vram)
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MODEL_BASENAME = None
