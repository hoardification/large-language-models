# Databricks notebook source
# MAGIC %pip install faiss-cpu==1.7.4 chromadb==0.3.21

# COMMAND ----------

# MAGIC %md
# MAGIC Use the data on <a href="https://newscatcherapi.com/" target="_blank">news topics collected by the NewsCatcher team</a>, who collect and index news articles and release them to the open-source community. The dataset can be downloaded from <a href="https://www.kaggle.com/kotartemiy/topic-labeled-news-dataset" target="_blank">Kaggle</a>.

# COMMAND ----------

import pandas as pd

pdf = pd.read_csv(f"{DA.paths.datasets}/news/labelled_newscatcher_dataset.csv", sep=";")
pdf["id"] = pdf.index
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC Vector libraries are [FAISS](https://faiss.ai/), [ScaNN](https://github.com/google-research/google-research/tree/master/scann), [ANNOY](https://github.com/spotify/annoy), and [HNSM](https://arxiv.org/abs/1603.09320).
# MAGIC
# MAGIC FAISS L2 (Euclidean distance), cosine similarity. (https://weaviate.io/blog/vector-library-vs-vector-database#feature-comparison---library-versus-database).

# COMMAND ----------

# MAGIC %md
# MAGIC Workflow of FAISS is captured in the diagram below. 
# MAGIC
# MAGIC <img src="https://miro.medium.com/v2/resize:fit:1400/0*ouf0eyQskPeGWIGm" width=700>

# COMMAND ----------

from sentence_transformers import InputExample

pdf_subset = pdf.head(1000)

def example_create_fn(doc1: pd.Series) -> InputExample:
    """
    Helper function that outputs a sentence_transformer guid, label, and text
    """
    return InputExample(texts=[doc1])

faiss_train_examples = pdf_subset.apply(
    lambda x: example_create_fn(x["title"]), axis=1
).tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC `Sentence-Transformers` [library](https://www.sbert.net/) language model to vectorize. Popular transformers on [Hugging Face Model Hub](https://huggingface.co/sentence-transformers).
# MAGIC `model = SentenceTransformer("all-MiniLM-L6-v2")`

# COMMAND ----------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2", 
    cache_folder=DA.paths.datasets
)  # Use a pre-cached model
faiss_title_embedding = model.encode(pdf_subset.title.values.tolist())
len(faiss_title_embedding), len(faiss_title_embedding[0])

# COMMAND ----------

# MAGIC %md
# MAGIC Save embedding vectors to FAISS index
# MAGIC Get Embeddings, normalize vectors, and add these vectors to the FAISS index. 

# COMMAND ----------

import numpy as np
import faiss

pdf_to_index = pdf_subset.set_index(["id"], drop=False)
id_index = np.array(pdf_to_index.id.values).flatten().astype("int")

content_encoded_normalized = faiss_title_embedding.copy()
faiss.normalize_L2(content_encoded_normalized)

# Index1DMap translates search results to IDs: https://faiss.ai/cpp_api/file/IndexIDMap_8h.html#_CPPv4I0EN5faiss18IndexIDMapTemplateE
# The IndexFlatIP below builds index
index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))
index_content.add_with_ids(content_encoded_normalized, id_index)

# COMMAND ----------

# MAGIC %md
# MAGIC Search for relevant documents

# COMMAND ----------

def search_content(query, pdf_to_index, k=3):
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)

    # We set k to limit the number of vectors we want to return
    top_k = index_content.search(query_vector, k)
    ids = top_k[1][0].tolist()
    similarities = top_k[0][0].tolist()
    results = pdf_to_index.loc[ids]
    results["similarities"] = similarities
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC Query for similar content! 

# COMMAND ----------

display(search_content("animal", pdf_to_index))

# COMMAND ----------

# MAGIC %md
# MAGIC Chromadb

# COMMAND ----------

import chromadb
from chromadb.config import Settings

chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DA.paths.user_db,  # this is an optional argument. If you don't supply this, the data will be ephemeral
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ChromaDB automatically loads a default embedding function, i.e. `SentenceTransformerEmbeddingFunction`. It can handle tokenization, embedding, and indexing automatically.

# COMMAND ----------

collection_name = "my_news"

# If you have created the collection before, you need to delete the collection first
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
    chroma_client.delete_collection(name=collection_name)

print(f"Creating collection: '{collection_name}'")
collection = chroma_client.create_collection(name=collection_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Add data to collection, Chroma can take care of text vectorization.

# COMMAND ----------

display(pdf_subset)

# COMMAND ----------

# MAGIC %md
# MAGIC Each document must have a unique id.

# COMMAND ----------

collection.add(
    documents=pdf_subset["title"][:100].tolist(),
    metadatas=[{"topic": topic} for topic in pdf_subset["topic"][:100].tolist()],
    ids=[f"id{x}" for x in range(100)],
)

# COMMAND ----------

# MAGIC %md
# MAGIC Find the 10 nearest neighbors

# COMMAND ----------

import json

results = collection.query(query_texts=["space"], n_results=10)

print(json.dumps(results, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Add filter statements. 

# COMMAND ----------

collection.query(query_texts=["space"], where={"topic": "SCIENCE"}, n_results=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus: Update data in a collection
# MAGIC Vector databases support changes to the data so we can update or delete data. 

# COMMAND ----------

collection.delete(ids=["id0"])

# COMMAND ----------

# MAGIC %md
# MAGIC No longer present.

# COMMAND ----------

collection.get(
    ids=["id0"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC Update a specific data point.

# COMMAND ----------

collection.get(
    ids=["id2"],
)

# COMMAND ----------

collection.update(
    ids=["id2"],
    metadatas=[{"topic": "TECHNOLOGY"}],
)

# COMMAND ----------

# MAGIC %md
# MAGIC Pass these documents as additional context for a language model to generate a response based on them. 

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=DA.paths.datasets)
lm_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=DA.paths.datasets)

pipe = pipeline(
    "text-generation",
    model=lm_model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    device_map="auto",
)

# COMMAND ----------

# MAGIC %md
# MAGIC Prompt engineering: 
# MAGIC - [Prompt engineering with OpenAI](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
# MAGIC - [GitHub repo that compiles best practices to interact with ChatGPT](https://github.com/f/awesome-chatgpt-prompts)

# COMMAND ----------

question = "What's the latest news on space development?"
context = " ".join([f"#{str(i)}" for i in results["documents"][0]])
prompt_template = f"Relevant context: {context}\n\n The user's question: {question}"

# COMMAND ----------

lm_response = pipe(prompt_template)
print(lm_response[0]["generated_text"])
