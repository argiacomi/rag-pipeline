import glob
import os
import subprocess
import uuid
from collections import namedtuple

import bm25s
import numpy as np
import Stemmer
from langchain_community.document_loaders import NotebookLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stemmer = Stemmer.Stemmer("english")


DocumentInfo = namedtuple("DocumentInfo", ["content", "metadata"])


def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(["git", "clone", github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False


def load_and_index_files(repo_path):
    exclude = [".git", "node_modules", "target"]

    filtered_paths = [
        d
        for d in glob.glob(f"{repo_path}/**/", recursive=True)
        if not any(excluded in d.split(os.sep) for excluded in exclude)
    ]
    notebook_paths = [
        os.path.dirname(d)
        for d in glob.glob(f"{repo_path}/**/*.ipynb", recursive=True)
        if not any(excluded in d.split(os.sep) for excluded in exclude)
    ]

    loaded_documents = list()
    file_type_counts = {}
    documents_dict = {}

    try:
        for path in filtered_paths:
            loader = GenericLoader.from_filesystem(
                path,
                glob="*.*",
                suffixes=[
                    ".css",
                    ".html",
                    ".js",
                    ".json",
                    ".jsp",
                    ".md",
                    ".rs",
                    ".sh",
                    ".ts",
                    ".tsx",
                ],
                exclude=["*.ipynb"],
                parser=LanguageParser(parser_threshold=6000),
            )
            loaded_documents.extend(loader.load())
        for path in notebook_paths:
            loader = NotebookLoader(
                str(repo_path),
                include_outputs=True,
                max_output_length=20,
                remove_newline=True,
            )

            loaded_documents.extend(loader.load())
        if loaded_documents:
            for ext in [
                ".css",
                ".html",
                ".ipynb",
                ".js",
                ".json",
                ".jsp",
                ".md",
                ".rs",
                ".sh",
                ".ts",
                ".tsx",
            ]:

                file_type_counts[ext] = len(
                    [f for f in loaded_documents if f.metadata["source"].endswith(ext)]
                )
            for doc in loaded_documents:
                file_path = doc.metadata["source"]
                relative_path = os.path.relpath(file_path, repo_path)
                file_id = str(uuid.uuid4())
                doc.metadata["source"] = relative_path
                doc.metadata["file_id"] = file_id
                documents_dict[file_id] = doc

    except Exception as e:
        print(f"Error loading files: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=100)

    split_documents = []
    for file_id, doc in documents_dict.items():
        split_docs = text_splitter.split_documents([doc])
        for i, doc in enumerate(split_docs):
            doc.metadata = doc.metadata.copy()
            doc.metadata["chunk_id"] = f"{file_id}_{i}"

        split_documents.extend(split_docs)

    corpus = [doc.page_content for doc in split_documents]
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever_save_name = "./retrievers/" + repo_path.split("/")[-1] + "_index_bm25"
    retriever.save(retriever_save_name, corpus=corpus)

    return (
        retriever_save_name,
        split_documents,
        file_type_counts,
        [doc.metadata["source"] for doc in split_documents],
    )


def search_documents(query, retriever_save_name, split_documents, top_k=5):
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    retriever = bm25s.BM25.load(retriever_save_name, load_corpus=True)

    # Get BM25 scores
    _, bm25_scores = retriever.retrieve(query_tokens, k=top_k)

    combined_scores = bm25_scores[0]

    # Get unique top documents
    unique_top_document_indices = list(set(combined_scores.argsort()[::-1]))[:top_k]

    return [split_documents[i] for i in unique_top_document_indices]
