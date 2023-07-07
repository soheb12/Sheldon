"""Main entrypoint for the app."""
import logging
import pickle
import uvicorn

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from schemas import ChatResponse
from query_data import get_results, init_client
import pandas as pd
from sentence_transformers import SentenceTransformer
from opensearchpy.helpers import bulk

load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
index_name = "knn-cosine-dense-vector-index"


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

    # TODO: initialize the openSearch client here
    # init_client()


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/context")
async def context_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = init_client()
    while True:
        try:
            context = await websocket.receive_text()
            print("context is : ")
            # print(context)
            await websocket.send_json({"context": context, "status": "processing"})

            dfs = pd.read_html(context)
            df = dfs[0]

            # drop the first row and select the next row for column names
            df = df.set_axis(df.iloc[0], axis=1)
            df = df.iloc[1:]

            # identify columns with NaN names
            nan_columns = df.columns[df.columns.isna()]
            # drop columns with NaN names
            df = df.drop(nan_columns, axis=1)

            df = df.dropna(axis=0)

            print(df.head())

            embedder = SentenceTransformer('all-mpnet-base-v2')

            vectors = []
            for index, row in df.iterrows():
                text = row['document_title'] + ":\n " + row['section_title'] + ":\n " + row['passage_text']
                vector = embedder.encode(text)
                vectors.append(vector)

            # Create a new dataframe with the vector embeddings and the original text
            df_vectors = pd.DataFrame({'vector': vectors, 'context': df['document_title'] + " :\n " + df['section_title'] + " :\n " + df['passage_text'], 'tags': df['passage_text']})

            # Convert dataframe to a list of dictionaries
            docs = df_vectors.to_dict('records')

            batch_size = 5
            for i in range(0, len(docs), batch_size):
                print("current batch")
                print(i)
                batch = docs[i:i+batch_size]
                bulk(client, batch, index=index_name)

            await websocket.send_json({"context": "Indexed documents", "status": "complete"})
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            await websocket.send_json({"context": "Encountered Error", "status": "failed"})

@app.websocket("/chat")
async def chat_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # question_handler = QuestionGenCallbackHandler(websocket)
    # stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=False)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Original get query result with langchain OrinConstruct a response
            # start_resp = ChatResponse(sender="bot", message="", type="start")
            # await websocket.send_json(start_resp.dict())

            # result = await qa_chain.acall(
            #     {"question": question, "chat_history": chat_history}
            # )
            # chat_history.append((question, result["answer"]))


            result_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(result_resp.dict())

            results = get_results(question)

            print("results being added")
            result_resp = ChatResponse(sender="bot", message=results, type="stream")
            await websocket.send_json(result_resp.dict())

            print("end_resp")
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
