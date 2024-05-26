import csv
import openai
import chromadb
from chromadb.utils import embedding_functions
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


from dotenv import load_dotenv

load_dotenv()

# # 1. Vectorise the sales response csv data


def generateVectorDatabase():
    # loader = CSVLoader(file_path="Train.csv")
    # documents = loader.load()

    # embeddings = OpenAIEmbeddings()
    # db = FAISS.from_documents(documents, embeddings)

    # return db

    # Load Data from files
    # Create Vector Database
    # Instantiate chromadb instance. Data is stored in memory only.
    # chroma_client = chromadb.Client()

    # Instantiate chromadb instance. Data is stored on disk (a folder named 'my_vectordb' will be created in the same folder as this file).
    chroma_client = chromadb.PersistentClient(path="my_vectordb")
    embedding_model = "all-mpnet-base-v2"
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model)
    try:
        print("TRY")
        print(chroma_client.list_collections())
        collection = chroma_client.get_collection(
            name="my_collection", embedding_function=sentence_transformer_ef)
        print(collection)
    except:

        with open('app/Train.csv') as file:
            lines = csv.reader(file)

            # Store the name of the menu items in this array. In Chroma, a "document" is a string i.e. name, sentence, paragraph, etc.
            documents = []

            # Store the corresponding menu item IDs in this array.
            metadatas = []

            # Each "document" needs a unique ID. This is like the primary key of a relational database. We'll start at 1 and increment from there.
            ids = []
            id = 1

            # Loop thru each line and populate the 3 arrays.
            for i, line in enumerate(lines):
                if i == 0:
                    # Skip the first row (the column headers)
                    continue

                documents.append(line[1])
                metadatas.append({"item_id": line[0]})
                ids.append(str(id))
                id += 1

        # Select the embedding model to use.
        # List of model names can be found here https://www.sbert.net/docs/pretrained_models.html
        # embedding_model = "all-MiniLM-L6-v2"

        # Create the collection, aka vector database. Or, if database already exist, then use it. Specify the model that we want to use to do the embedding.
        collection = chroma_client.get_or_create_collection(
            name="my_collection", embedding_function=sentence_transformer_ef)

        # Add all the data to the vector database. ChromaDB automatically converts and stores the text as vector embeddings. This may take a few minutes.
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    return collection
    # Testing the vector database
    # # Query mispelled word: 'vermiceli'. Expect to find the correctly spelled 'vermicelli' item
    # results = collection.query(
    #     query_texts=["What is main programming language?"],
    #     n_results=3,
    #     include=['documents', 'distances', 'metadatas']
    # )
    # print(results['documents'])


# OPENAI LLM
def setupOpenAI():
    # embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    return llm


def setupLangchain(llm):
    template = """
    You are Personal Assistant to Nilkumar Patel.
    I will share a user question with you and you will give me the best answer that
    I should send to this user based on provided information,

    Below is a question I received from the user:
    {question}

    Here is a information related to user question:
    {context}

    follow the below given rules:
    1. Always answer in first person.
    2. Limit the answers to 150 words.
    3. If the information is not relavent to question then suggest use to ask question related to professional life.

    Please write the best response within 150 words:
    """
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    return chain


# def retrieve_info(query, db):
#     similar_response = db.similarity_search(query, k=3)

#     page_contents_array = [doc.page_content for doc in similar_response]

#     # print(page_contents_array)

#     return page_contents_array
def retrieve_info(question, collection):
    # Query mispelled word: 'vermiceli'. Expect to find the correctly spelled 'vermicelli' item
    results = collection.query(
        query_texts=[question],
        n_results=5,
        include=['documents', 'distances', 'metadatas']
    )

    # print(results['documents'])
    print("-----------------------------------")
    return results['documents']


# 4. Retrieval augmented generation
def generate_response(question, chain, collection):
    context = retrieve_info(question, collection)
    response = chain.run(question=question, context=context)
    return response

# 5. Build an app with streamlit


def main(question, chain, collection):
    # question = "What is your favirote place to go?"
    answer = generate_response(question, chain, collection)
    print(answer)
    return answer


# if __name__ == "__main__":

#     print("Ask Your Question: ")
#     question = input()
#     if len(question) > 0:
#         collection = generateVectorDatabase()
#         llm = setupOpenAI()
#         chain = setupLangchain(llm=llm)

#         continue_ = True
#         while True:
#             main(question, collection=collection, chain=chain)
#             print("have a another question?: [y/n]")
#             continue_ = True if input() == "y" else False
#             if continue_:
#                 print("Ask Your Question: ")
#                 question = input()
#                 continue
#             else:
#                 break
