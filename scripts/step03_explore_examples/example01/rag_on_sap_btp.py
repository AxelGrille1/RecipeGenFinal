# from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
# from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
# from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
# from library.constants.folders import FILE_ENV
# from library.constants.table_names import TABLE_NAME
# from library.util.logging import initLogger
# from langchain_community.vectorstores.hanavector import HanaDB
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from library.data.hana_db import get_connection_to_hana_db
# from dotenv import load_dotenv
# import logging
# from firebase_admin import credentials, initialize_app, firestore
# import json
# import random

# log = logging.getLogger(__name__)
# initLogger()


# def main(input, uuid):
#     # def main():
#     # Load environment variables
#     load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

#     # Get the connection to the HANA DB
#     connection_to_hana = get_connection_to_hana_db()
#     log.info("Connection to HANA DB established")

#     # Get the proxy client for the AI Core service
#     proxy_client = get_proxy_client("gen-ai-hub")

#     # Create the OpenAIEmbeddings object
#     embeddings = OpenAIEmbeddings(
#         proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
#     )
#     log.info("OpenAIEmbeddings object created")

#     llm = ChatOpenAI(proxy_model_name="gpt-35-turbo", proxy_client=proxy_client)
#     log.info("ChatOpenAI object created")

#     # Create a memory instance to store the conversation history
#     memory = ConversationBufferMemory(
#         memory_key="chat_history", output_key="answer", return_messages=True
#     )
#     log.info("Memory object created")

#     # Create the HanaDB object
#     db = HanaDB(
#         embedding=embeddings, connection=connection_to_hana, table_name=TABLE_NAME
#     )

#     # -------------------------------------------------------------------------------------
#     # Fetch the data from the HANA DB and use it to answer the question using the
#     # best 5 matching information chunks
#     # -------------------------------------------------------------------------------------

#     # Create a retriever instance of the vector store
#     retriever = db.as_retriever(search_kwargs={"k": 5})
#     log.info("Retriever instance of the vector store created")

#     # -------------------------------------------------------------------------------------
#     # Call the LLM with the RAG information retrieved from the HANA DB
#     # -------------------------------------------------------------------------------------

#     # Create prompt template
#     prompt_template = """
#     You are a recipe generator. Generate similar recipes using existing ones.
#     Scale the recipes for 100 Kg or 100L. Use the metric system. Give all the steps on an industrial way.
#     Use the following pieces of context to answer the question at the end.
#     Your answer will be formatted as json files containing the following fields: UUID, allergens, ingredients, nutrients, price, recipe, recipe_name, sustainability_index.
#     You will let the uuid empty at this step, and randomly assign a sustainability index between 1 and 5.
#     Randomly assign a price and a numeric value for nutrients (carbohydrates, fats, proteins) in a single string.
#     All the fields are strings.
#     Generate a single recipe.
#     Create an original name for the recipe's name, choose marketable names, no industrial names.
#     Please use sugar as sweetener, fruit content shall be 35% in the fruit preparation (split into 15% whole raspberries or raspberry pieces and 20% via raspperry juice concentrate). Legislation area is EU. Product should be halal and kosher suitable.
#     Fruit preparation shall be produced in Oosterhout. Please develop one version without flavour and colour and one version with natural flavour and colour".

#     ```
#     {context}
#     ```

#     Question: {question}
#     """
#     # Please develop a fruit preparation for spoonable yoghurt. Please use sugar as sweetener, fruit content shall be 35% in the fruit preparation (split into 15% whole raspberries or raspberry pieces and 20% via raspperry juice concentrate), legislation area is EU, product should be halal and kosher suitable. Furthermore, fruit preparation shall be produced in Oosterhout. Please develop one version without flavour and colour and one version with natural flavour and colour
#     PROMPT = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )
#     chain_type_kwargs = {"prompt": PROMPT}

#     # Create a conversational retrieval chain
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm,
#         retriever,
#         return_source_documents=True,
#         memory=memory,
#         verbose=False,
#         combine_docs_chain_kwargs=chain_type_kwargs,
#     )

#     # -------------------------------------------------------------------------------------
#     # Provide the response to the user
#     # -------------------------------------------------------------------------------------

#     log.header("Welcome to RecipeGen! Type 'exit' to end the session")

#     while True:
#         # Prompt the user for a question
#         # question = input("Recipe to generate: ")
#         question = f"generate a recipe of {input}, fill the uuid field with {uuid}"

#         # Check if the user wants to exit
#         if question.lower() == "exit":
#             print("Goodbye!")
#             break

#         log.info(
#             f"Asking a question: {question}",
#         )

#         # Invoke the conversational retrieval chain with the user's question
#         result = qa_chain.invoke({"question": question})

#         # Output the answer from LLM
#         # log.success("Answer from LLM:")
#         # print(result["answer"])
#         json_result = result["answer"]

#         # Output the source document chunks used for the answer
#         source_docs = result["source_documents"]

#         print("================")
#         log.info(f"Number of used source document chunks: {len(source_docs)}")

#         content = []
#         for doc in source_docs:
#             print("-" * 80)
#             log.info(doc.page_content)
#             # log.info(doc.metadata)
#             content.append(doc.page_content)

#         return content, json_result


# def init_firebase():
#     service_account_key_file = (
#         "scripts/step03_explore_examples/example01/firebase_credentials.json"
#     )
#     cred = credentials.Certificate(service_account_key_file)
#     app = initialize_app(
#         cred,
#         options={
#             "databaseURL": "https://recipegen-305f4-default-rtdb.europe-west1.firebasedatabase.app/"
#         },
#     )
#     return app


# def push_to_database(json_article):
#     db = firestore.client()
#     articles_ref = db.collection("Outputs")
#     data = json.loads(json_article)
#     try:
#         result = articles_ref.add(data)
#         print("Document added successfully.")
#     except json.JSONDecodeError as e:
#         print("Error decoding JSON:", e)
#     except Exception as e:
#         print("Error:", e)
#         return result


# def get_recipe_name():
#     db = firestore.client()
#     ref = db.collection("Inputs")
#     documents = ref.stream()
#     for doc in documents:
#         return doc.get("recipe_name")


# def get_uuid():
#     db = firestore.client()
#     ref = db.collection("Inputs")
#     documents = ref.stream()
#     for doc in documents:
#         return doc.get("UUID")


# def transform_json(json_file, sources):
#     # print("json file: ", json_file)
#     # json_file = json_file[0]
#     # json_file = json_file.replace("json)", "")
#     # json_file = json_file.replace("```", "")

#     # print("json file modified: ", json_file)

#     if isinstance(json_file, str):
#         recipe_dict = json.loads(json_file)
#     else:
#         json_article = json_file[0]
#         recipe_dict = json.loads(json_article)

#     price = round(random.uniform(5, 7), 2) * 100
#     recipe_dict["price"] = f"${price}"

#     # recipe_dict["ingredients"] = recipe_dict["ingredients"].replace(",", "\n")

#     recipe_dict["sources"] = str(sources)
#     # recipe_dict["sources"] = recipe_dict["sources"].lower().replace("chunk", "recipe")

#     sustainability_score = int(recipe_dict["sustainability_index"])
#     score = "\u2605" * sustainability_score + "\u2606" * (5 - sustainability_score)
#     recipe_dict["sustainability_index"] = score

#     # recipe_dict["nutrients"] = recipe_dict["nutrients"].replace(",", "\n")
#     return json.dumps(recipe_dict, indent=4)


# def parse_json(json_file):
#     if json_file.startswith("```json") and json_file.endswith("```"):
#         json_content = json_file[len("```json") : -len("```")]
#         # print("parsed json: ", json_content)
#         return json_content
#     else:
#         return json_file


# def get_data():
#     input_recipe = get_recipe_name()
#     uuid = get_uuid()
#     return input_recipe, uuid


# if __name__ == "__main__":
#     # Init the database and get values from it
#     init_firebase()

#     input, uuid = get_data()

#     for _ in range(3):
#         # Execute the RAG
#         sources, json_result = main(input, uuid)

#         # print("Raw Json", json_result)
#         # Transform data to make it fancier
#         transformed_json = transform_json(json_result, sources)
#         parsed_json = parse_json(transformed_json)

#         print("Answer from LLM:\n", parsed_json)

#         # Push to database
#         push_to_database(parsed_json)


from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from library.constants.folders import FILE_ENV
from library.constants.table_names import TABLE_NAME
from library.util.logging import initLogger
from langchain_community.vectorstores.hanavector import HanaDB
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from library.data.hana_db import get_connection_to_hana_db
from dotenv import load_dotenv
import logging
from firebase_admin import credentials, initialize_app, firestore
import json
import random

log = logging.getLogger(__name__)
initLogger()


def main(input, uuid):
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

    # Get the connection to the HANA DB
    connection_to_hana = get_connection_to_hana_db()
    log.info("Connection to HANA DB established")

    # Get the proxy client for the AI Core service
    proxy_client = get_proxy_client("gen-ai-hub")

    # Create the OpenAIEmbeddings object
    embeddings = OpenAIEmbeddings(
        proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
    )
    log.info("OpenAIEmbeddings object created")

    llm = ChatOpenAI(proxy_model_name="gpt-35-turbo", proxy_client=proxy_client)
    log.info("ChatOpenAI object created")

    # Create a memory instance to store the conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )
    log.info("Memory object created")

    # Create the HanaDB object
    db = HanaDB(
        embedding=embeddings, connection=connection_to_hana, table_name=TABLE_NAME
    )

    # Create a retriever instance of the vector store
    retriever = db.as_retriever(search_kwargs={"k": 3})
    log.info("Retriever instance of the vector store created")

    # Create prompt template
    prompt_template = """
    You are a recipe generator. Generate similar recipes using existing ones.
    Scale the recipes for 100 Kg or 100L. Use the metric system. Give all the steps on an industrial way.
    Use the following pieces of context to answer the question at the end.
    Your answer will be formatted as json files containing the following fields: UUID, allergens, ingredients, nutrients, price, recipe, recipe_name, sustainability_index.
    You will let the uuid empty at this step, and randomly assign a sustainability index between 1 and 5.
    Randomly assign a price and a numeric value for nutrients (carbohydrates, fats, proteins) in a single string.
    All the fields are strings.
    Generate a single recipe.
    Create an original name for the recipe's name, choose marketable names, no industrial names.
    
    ```
    {context}
    ```

    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    # Create a conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True,
        memory=memory,
        verbose=False,
        combine_docs_chain_kwargs=chain_type_kwargs,
    )

    log.header("Welcome to RecipeGen! Type 'exit' to end the session")

    while True:
        question = f"generate a recipe of {input}, fill the uuid field with {uuid}"

        if question.lower() == "exit":
            print("Goodbye!")
            break

        log.info(f"Asking a question: {question}")

        try:
            result = qa_chain.invoke({"question": question})
            json_result = result["answer"]
            source_docs = result["source_documents"]

            log.info(f"Number of used source document chunks: {len(source_docs)}")

            content = [doc.page_content for doc in source_docs]
            return content, json_result

        except Exception as e:
            log.error(f"Error during question processing: {e}")
            return [], "{}"


def init_firebase():
    service_account_key_file = (
        "scripts/step03_explore_examples/example01/firebase_credentials.json"
    )
    cred = credentials.Certificate(service_account_key_file)
    app = initialize_app(
        cred,
        options={
            "databaseURL": "https://recipegen-305f4-default-rtdb.europe-west1.firebasedatabase.app/"
        },
    )
    return app


def push_to_database(json_article):
    db = firestore.client()
    articles_ref = db.collection("Outputs")
    try:
        data = json.loads(json_article)
        result = articles_ref.add(data)
        print("Document added successfully.")
        return result
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
    except Exception as e:
        print("Error:", e)
        return None


def get_recipe_name():
    db = firestore.client()
    ref = db.collection("Inputs")
    documents = ref.stream()
    for doc in documents:
        return doc.get("recipe_name")


def get_uuid():
    db = firestore.client()
    ref = db.collection("Inputs")
    documents = ref.stream()
    for doc in documents:
        return doc.get("UUID")


def transform_json(json_file, sources):
    print("def transform_json: ", json_file)
    if not json_file:
        log.error("Empty JSON data received")
        return "{}"

    try:
        recipe_dict = json.loads(json_file)
    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON: {e}")
        return "{}"

    price = round(random.uniform(5, 7), 2) * 100
    recipe_dict["price"] = f"${price}"
    recipe_dict["sources"] = str(sources)

    sustainability_score = int(recipe_dict["sustainability_index"])
    score = "\u2605" * sustainability_score + "\u2606" * (5 - sustainability_score)
    recipe_dict["sustainability_index"] = score
    print("recipe_dict: ", recipe_dict)

    return json.dumps(recipe_dict, indent=4)


def parse_json(json_file):
    print("json file", json_file)
    if json_file.startswith("```json") and json_file.endswith("```"):
        json_content = json_file[len("```json") : -len("```")]
        return json_content
    return json_file


def get_data():
    input_recipe = get_recipe_name()
    uuid = get_uuid()
    return input_recipe, uuid


if __name__ == "__main__":
    init_firebase()

    input, uuid = get_data()

    for _ in range(3):
        sources, json_result = main(input, uuid)
        parsed_json = parse_json(json_result)

        transformed_json = transform_json(parsed_json, sources)

        print("Answer from LLM:\n", transformed_json)

        push_to_database(transformed_json)
