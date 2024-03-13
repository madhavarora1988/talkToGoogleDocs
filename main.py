import os
import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
import pandas as pd
from langchain_anthropic import ChatAnthropic
from metrics.evaluate_flesch_kincaid_score import flesch_kincaid_grade_level

# openai api key
os.environ["OPENAI_API_KEY"] = "<OpenAI Key>"
os.environ[
    "ANTHROPIC_API_KEY"] = "<AnthropicAI Key>"


def load_youtube_video(video_url):
    loader = YoutubeLoader.from_youtube_url(
        video_url, add_video_info=True
    )
    data = loader.load()
    return data


def persist_data_in_chroma(persist_directory, data):
    # various supported model options are available
    embedding_function = OpenAIEmbeddings()

    # Chroma instance takes in a directory location, embedding function and ids-> list of documentIds
    chroma_db = Chroma.from_documents(data,
                                      embedding_function, persist_directory)

    return chroma_db


def create_llm(model_name, **kwargs):
    # Map model names to their corresponding class constructors
    model_map = {
        "claude-3-sonnet-20240229": ChatAnthropic,
        "gpt-3.5-turbo": ChatOpenAI
        # Add more models here
    }
    # Get the constructor for the given model_name or raise a ValueError if not found
    model_constructor = model_map.get(model_name)
    if model_constructor:
        return model_constructor(model_name=model_name, **kwargs)
    else:
        raise ValueError("Unsupported model_name")


def summarize_data(db, llm):
    print("summarizing data with llm : ", llm)

    chatbot_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1})
    )

    template = """
    respond as clearly as possible {query}?
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )

    summary = chatbot_chain.run(
        prompt.format(query="summarize the video transcript in 100 words?")
    )
    return summary


def read_excel(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    youtube_links = df['Youtube Links'].tolist()
    return youtube_links


def update_excel_row(file_path, row_index, title, description, model_name):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Check if row_index is valid
    if row_index >= len(df):
        raise IndexError("The row_index is out of bounds of the Excel file")

    # Update the 'Title' and 'Description' columns for the specified row
    df.at[row_index, 'Model'] = model_name
    df.at[row_index, 'Title'] = title
    df.at[row_index, 'Description'] = description

    # Write the updated DataFrame back to the Excel file
    df.to_excel(file_path, index=False)

def process_args():
    parser = argparse.ArgumentParser(description="Process model_name and file_path values")
    parser.add_argument('model_name', type=str, help='The name of the model to use')
    parser.add_argument('--file_path', type=str, default="YoutubeSummaries.xlsx",
                        help='The path to the Excel file with YouTube summaries')
    return parser

def extract_title(data):
    try:
        title = data[0].metadata.get('title', "") if data else ""
    except (IndexError, AttributeError, KeyError):
        title = ""
    return title


if __name__ == '__main__':
    args = process_args().parse_args()
    file_path = args.file_path
    model_name = args.model_name

    links = read_excel(file_path)
    response_list = []
    print(links)
    for index, link in enumerate(links):
        data = load_youtube_video(link)
        title = extract_title(data)
        db = persist_data_in_chroma('db', data)

        llm = create_llm(model_name, temperature=0, max_tokens=500)
        summary = summarize_data(db, llm)
        flesch_kincaid_score = flesch_kincaid_grade_level(summary)

        json_resp = {"index": index,
                     "title": title,
                     "summary": summary,
                     "flesch_kincaid_score": flesch_kincaid_score
                     }
        response_list.append(json_resp)

        update_excel_row(file_path, index, title, summary, model_name)
    print('response from model: ', response_list)
