import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
import pandas as pd

# openai api key
os.environ["OPENAI_API_KEY"] = "<Open AI key here>"
# openai api key from env
openai_api_key = os.environ.get('OPENAI_API_KEY')


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


def summarize_data(db):
    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo", max_tokens=500
    )

    chatbot_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo-16k", max_tokens=500
        ),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 12})
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

def update_excel_row(file_path, row_index, title, description):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Check if row_index is valid
    if row_index >= len(df):
        raise IndexError("The row_index is out of bounds of the Excel file")

    # Update the 'Title' and 'Description' columns for the specified row
    df.at[row_index, 'Title'] = title
    df.at[row_index, 'Description'] = description

    # Write the updated DataFrame back to the Excel file
    df.to_excel(file_path, index=False)



if __name__ == '__main__':
    file_path = "YoutubeSummaries.xlsx"
    links = read_excel(file_path)
    print(links)
    for index, link in enumerate(links):
        data = load_youtube_video(link)
        title = data[0].metadata['title'] if data else ""
        db = persist_data_in_chroma('db', data)
        summary = summarize_data(db)
        update_excel_row(file_path, index, title, summary)


