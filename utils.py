import os
import re


def format_documents(documents):
    numbered_docs = "\n".join(
        [
            f"{i+1}. {os.path.basename(doc.metadata['source'])}: {doc.page_content}"
            for i, doc in enumerate(documents)
        ]
    )
    return numbered_docs
