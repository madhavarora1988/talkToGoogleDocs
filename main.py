import os
import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_googledrive.document_loaders import GoogleDriveLoader

os.environ["OPENAI_API_KEY"] = "<OpenAI key>"
os.environ[
    "ANTHROPIC_API_KEY"] = "<AnthropicAI Key>"


def load_google_drive_data(folder_id):
    loader = GoogleDriveLoader(
        # provide the path to the credentials file, I have it in the project root.
        gdrive_api_file="credentials.json",
        folder_id=folder_id,
        # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
        recursive=False
    )
    loaded_data = loader.load()
    print("loaded data ", loaded_data)
    return loaded_data


def persist_data_in_chroma(persist_directory, data):
    # various supported model options are available
    embedding_function = OpenAIEmbeddings()
    print('data: ', data)

    # Chroma instance takes in a directory location, embedding function and ids-> list of documentIds
    chroma_db = Chroma.from_documents(data,
                                      embedding_function, persist_directory)

    return chroma_db


def create_llm(model_name, **kwargs):
    # Map model names to their corresponding class constructors
    model_map = {
        "claude-3-sonnet-20240229": ChatAnthropic,
        "gpt-3.5-turbo": ChatOpenAI,
        "gpt-4o": ChatOpenAI,
        # Add more models here
    }
    # Get the constructor for the given model_name or raise a ValueError if not found
    model_constructor = model_map.get(model_name)
    if model_constructor:
        return model_constructor(model_name=model_name, **kwargs)
    else:
        raise ValueError("Unsupported model_name")


def query_data(db, llm, query):
    print("summarizing data with llm : ", llm)

    chatbot_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1})
    )

    template = """
    respond as clearly as possible and in max 200 words {query}?
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )

    summary = chatbot_chain.run(
        prompt.format(query=query)
    )
    return summary







def process_args():
    parser = argparse.ArgumentParser(description="Process model_name")
    parser.add_argument('--model_name', type=str, help='The name of the model to use')
    return parser


if __name__ == '__main__':
    args = process_args().parse_args()
    model_name = args.model_name
    folder_id = '<Google Drive Folder Id>'

    # Load the data from Google Drive
    data = load_google_drive_data(folder_id)
    # Persist the data in Chroma
    db = persist_data_in_chroma('db', data)
    # configure the respective model
    llm = create_llm(model_name, temperature=0, max_tokens=500)
    # TODO: Persist the db, and use that instance to query
    response = query_data(db, llm, "tell me about the TimeBench, and provide the path to the google drive document?")

    print('response from model: ', response)
