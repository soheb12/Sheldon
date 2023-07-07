"""Create a ChatVectorDBChain for question/answering."""
import os

from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStore
from pydantic import Field
import time

doc_template = """--- document start ---
content:{page_content}
--- document end ---
"""

CONFLUENT_DOC_PROMPT = PromptTemplate(
    template=doc_template, input_variables=["page_content"]
)

prompt_template = """You are an AI assistant for Confluent and Kafka documentation. 
You are given the following extracted parts of a long document and a question. Your task is to answer the question the best you can. Pretend you are a human answering the question.
The docs may not have an exact answer to the question, but you should try to answer the question as best you can. Your job is to help the user find the answer to the question.
If the question includes a request for code, provide a fenced code block directly from the documentation.
Question: {question}
Documents:
=========
{context}
=========
Answer in Markdown:"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ChatVectorDBChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        model_name="gpt-3.5-turbo",
        max_tokens=520,
        temperature=0.1,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )
    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=QA_PROMPT,
        document_prompt=CONFLUENT_DOC_PROMPT,
        callback_manager=manager,
    )

    qa = ChatVectorDBChain(
        vectorstore=vectorstore,
        verbose=True,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        top_k_docs_for_context=6,
    )

    return qa








import boto3
import numpy as np
import openai
import json
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

embedder = SentenceTransformer('all-mpnet-base-v2')
prompt = ""
client = None
index_name = "knn-cosine-dense-vector-index"
index_body = {
    "settings":{
        "index": {
            "number_of_shards": 20,
            "knn": {
                "algo_param": {
                    "ef_search": 512,
                    "ef_construction": 512,
                    "m": 16
                }
            },
            "knn": True 
        },
    },
    "mappings": {
        "properties": {
            "vector": {
                "type": "knn_vector", 
                "dimension": 768,
                "index": True,
                "similarity": "cosine",
                "method": {
                  "name": "hnsw",
                  "space_type": "cosinesimil",
                  "engine": "nmslib",
                  "parameters": {
                    "ef_construction": 512,
                    "m": 16
                   }
                }
            },
            "context": {"type": "text"},
            "tags": {"type": "text"}
        }
    }
}

def init_client():
    host = 'vpc-context-storage-ngiys3dwxhozwlvlumxrtoorsu.us-east-1.es.amazonaws.com'
    region = 'us-east-1'
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region)

    host = 'search-sheldon-demo-hb45yvdjp5id6x5d4hlvan7fve.us-east-1.es.amazonaws.com'
    region = 'us-east-1'
    auth = ('sheldon', 'Sheldon@123')

    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20
    )

    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=index_body)

    # client = OpenSearch(
    #             hosts=[{"host": host, "port": 443}],
    #             http_auth=auth,
    #             use_ssl=True,
    #             verify_certs=True,
    #             connection_class=RequestsHttpConnection,
    # )

    return client


def get_context(question):
    global embedder
    client = init_client()
    input_vector = embedder.encode(question)
    query ={
        "size": 5,
        "query":{
            "bool":{
                "must":[
                    {
                    "knn":{
                        "vector":{
                            "vector":input_vector,
                            "k":5
                        }
                    },

                    }
                ]
            }
        }
    }

    result = client.search(index=index_name, body=query)
    hits = result['hits']['hits']

    count = 1
    context = ""
    for hit in hits:
        print(str(count) + ". " + f"Document ID: {hit['_id']}, Relevance Score: {hit['_score']}")
        count = count + 1
        vec_res = hit['_source']['vector']
        res = hit['_source']['context']
        print(res)
        context = context + res
    return context


def get_answer(question, context):
    global prompt
    openai.api_key = "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"
    question = question
    prompt = """
    Based on the context answer the below question:\n
    Context: 
    """ + context + "\n Answer the question accurately from context above: \n Q: " 
    prompt = prompt + question + "\n A: "

    print("prompt is : **")
    print(prompt)

    response = openai.Completion.create(
        model =  "text-davinci-003",
        prompt =  prompt,
        temperature =  0.7,
        max_tokens = 512,
        top_p = 1,
        timeout = 5
    )
    answer = response["choices"][0]["text"].strip()
    prompt = prompt + answer + "\n\n Q: "
    return answer


def get_results(question):
    context = get_context(question)
    answer = get_answer(question, context)
    return answer
    # time.sleep(3)
    # return "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."