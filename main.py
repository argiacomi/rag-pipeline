import os
import re
import tempfile

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import GREEN, RESET_COLOR, WHITE, model_name
from file_processing import clone_github_repo, load_and_index_files
from questions import QuestionContext, ask_question

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    github_url = input("Enter the GitHub URL of the repository: ")
    repo_name = github_url.split("/")[-1]
    print("Cloning the repository...")
    with tempfile.TemporaryDirectory() as local_path:
        if clone_github_repo(github_url, local_path):
            retriever, documents, file_type_counts, filenames = load_and_index_files(
                local_path
            )
            if retriever is None:
                print("No documents were found to index. Exiting.")
                exit()

        print("Repository cloned. Indexing files...")

        llm = ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        template = """You are a Software Engineering expert whose job is to answer questions related to {repo_name} ({github_url})
        Some relevant background info on the repository:
            - Conv: {conversation_history}
            - Docs: {numbered_documents}
            - FileCount: {file_type_counts}
            - FileNames: {filenames}

        Your Instructions:
        1. Answer based on the context & documents provided
        2. Focus on the repository & it's codebase
        4. Unsure? Say "I am not sure"
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("human", "{question}"),
            ]
        )

        llm_chain = prompt | llm

        conversation_history = ""
        question_context = QuestionContext(
            retriever,
            documents,
            llm_chain,
            repo_name,
            github_url,
            conversation_history,
            file_type_counts,
            filenames,
        )
        while True:
            try:
                user_question = input(
                    "\n"
                    + WHITE
                    + "Ask a question about the repository (type 'exit()' to quit): "
                    + RESET_COLOR
                )
                if user_question.lower() == "exit()":
                    break
                print("Thinking...")
                user_question = re.sub(r"\s+", " ", user_question).strip()

                answer = ask_question(user_question, question_context)
                print(GREEN + "\nANSWER\n" + answer + RESET_COLOR + "\n")
                conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        else:
            print("Failed to clone the repository.")
