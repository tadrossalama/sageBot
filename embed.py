from pathlib import Path
import os
import sys
import faiss
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI, LLMChain


trainingData = list(Path("training/facts/").glob("**/*.*"))

data = []
for training in trainingData:
  with open(training) as f:
    print(f"Add {f.name} to dataset")
    data.append(f.read())

textSplitter = CharacterTextSplitter(chunk_size=2000, separator="\n")

docs = []
for sets in data:
  docs.extend(textSplitter.split_text(sets))

store = FAISS.from_texts(docs, OpenAIEmbeddings())
faiss.write_index(store.index, "training.index")
store.index = None

with open("faiss.pkl", "wb") as f:
  pickle.dump(store, f)
