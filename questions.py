# questions.py
from file_processing import search_documents
from utils import format_documents


class QuestionContext:

    def __init__(
        self,
        retriever,
        documents,
        llm_chain,
        repo_name,
        github_url,
        conversation_history,
        file_type_counts,
        filenames,
    ):
        self.retriever = retriever
        self.documents = documents
        self.llm_chain = llm_chain
        self.repo_name = repo_name
        self.github_url = github_url
        self.conversation_history = conversation_history
        self.file_type_counts = file_type_counts
        self.filenames = filenames


def ask_question(question, context: QuestionContext):
    relevant_docs = search_documents(
        question, context.retriever, context.documents, top_k=5
    )

    numbered_documents = format_documents(relevant_docs)
    question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{numbered_documents}.\n\nThe question is {question}."

    answer_with_sources = context.llm_chain.invoke(
        {
            "question": question_context,
            "repo_name": context.repo_name,
            "github_url": context.github_url,
            "conversation_history": context.conversation_history,
            "numbered_documents": numbered_documents,
            "file_type_counts": context.file_type_counts,
            "filenames": context.filenames,
        }
    )
    return answer_with_sources.content
