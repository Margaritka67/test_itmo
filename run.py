from typing import List
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
import os
import uuid

from dotenv import load_dotenv
load_dotenv(override=True)

DB_COLLECTION_NAME="itmo"
URLS = ["https://abit.itmo.ru/program/master/ai", "https://abit.itmo.ru/program/master/ai_product"]

CHAT_API_KEY = os.getenv("CHAT_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
CHAT_API_BASE = os.getenv("CHAT_API_BASE")

QDRANT_URL = os.getenv("QDRANT_URL")

EMBED_MODEL = os.getenv("EMBED_MODEL")
EMBED_API_BASE = os.getenv("EMBED_API_BASE")
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
EMBED_SIZE = str(os.getenv("EMBED_SIZE"))
    
LLM_CLIENT = ChatOpenAI(
    model_name=CHAT_MODEL,
    openai_api_key=CHAT_API_KEY,
    openai_api_base=CHAT_API_BASE,
    temperature=0.05,
)

EMBED_CLIENT = OpenAI(
    api_key=EMBED_API_KEY,
    base_url=EMBED_API_BASE,
)

QDRANT_CLIENT = QdrantClient(url=QDRANT_URL)

async def completion(prompt: str) -> str:
	response = await LLM_CLIENT.ainvoke(
	    input=[
		{
		    "role": "system",
		    "content": "You are an AI assistant to consalt users about ITMO masters programms.",
		},
		{"role": "user", "content": prompt},
	    ],
	)
	return response.content.strip()


def embed(text: str) -> List[List[float]]:
    if len(text) == 0:
        return []

    res = EMBED_CLIENT.embeddings.create(
        input=text,
        model=EMBED_MODEL,
        encoding_format="float",
    )
    return [item.embedding for item in res.data][0]


def create_db():
	if not QDRANT_CLIENT.collection_exists(collection_name=DB_COLLECTION_NAME):
		QDRANT_CLIENT.create_collection(
			collection_name=DB_COLLECTION_NAME,
			vectors_config=VectorParams(size=EMBED_SIZE, distance=Distance.COSINE),
		)

		#Парсинг данных
		points = []
		for url in URLS:
			elements = partition_html(url=url)
			chunks = chunk_by_title(elements, max_characters=2000, overlap=200)
			for chunk in chunks:
				point=PointStruct(
					id=str(uuid.uuid4()),
					vector=embed(chunk.text.strip()),
					payload={"page_content": chunk.text, "url": url}
				)
				points.append(point)

		QDRANT_CLIENT.upsert(
			collection_name=DB_COLLECTION_NAME,
			points=points
		)
		print("Индекс успешно создан!")
	else:
		print("Индекс был создан ранее.")

def main():
	create_db()

	system_prompt = (
		"Use the given context and not your previous knowledge to answer the question. "
		"If you don't know the answer, say you don't know. "
		"Context: {context}"
	)

	prompt = ChatPromptTemplate.from_messages(
		[
			("system", system_prompt),
			("human", "{input}"),
		]
	)

	vector_store = QdrantVectorStore(
		client=QDRANT_CLIENT,
		collection_name=DB_COLLECTION_NAME,
		embedding=OpenAIEmbeddings(
			openai_api_key=EMBED_API_KEY,
			openai_api_base=EMBED_API_BASE,
			model=EMBED_MODEL,
		),
	)

	retriever = vector_store.as_retriever(search_kwargs={"k": 10})

	question_answer_chain = create_stuff_documents_chain(LLM_CLIENT, prompt)
	chain = create_retrieval_chain(retriever, question_answer_chain)

	#Интерактивный цикл вопрос-ответ
	print("RAG система готова. Задавайте вопросы (для выхода введите 'exit'):")
	while True:
		query = input("\nВаш вопрос: ")
		if query.lower() == 'exit':
			break
			
		result = chain.invoke({"input": query})
		print(f"\nОтвет: {result['answer']}")

if __name__ == "__main__":
    main()
    
    

