from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from transformers import AutoTokenizer
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, AsyncGenerator
from langchain_openai import ChatOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import os
import uuid

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

CHAT_API_KEY = os.getenv("CHAT_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
CHAT_API_BASE = os.getenv("CHAT_API_BASE")

QDRANT_URL = os.getenv("QDRANT_URL")

EMBED_MODEL = os.getenv("EMBED_MODEL")
EMBED_API_BASE = os.getenv("EMBED_API_BASE")
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
EMBED_SIZE = os.getenv("EMBED_SIZE")

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))
    
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

QUADRANT_CLIENT = AsyncQdrantClient(url=QDRANT_URL, timeout=10)

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


def embed(text: List[str]) -> List[List[float]]:
    if len(text) == 0:
        return []
    res = EMBED_CLIENT.embeddings.create(
        input=text,
        model=EMBED_MODEL,
        encoding_format="float",
    )
    return [item.embedding for item in res.data]


async def main():

	#Парсинг данных
	urls = ["https://abit.itmo.ru/program/master/ai", "https://abit.itmo.ru/program/master/ai_product"]

	
	points = []
	for url in urls:
		elements = partition_html(url=url)
		chunks = chunk_by_title(elements, max_characters=5000, overlap=500)
        
		for c in chunks:
			row = {}
			row['filename'] = c.metadata.filename
			row['text'] = c.text
			row['url'] = url
			point=PointStruct(
				id=uuid.uuid4(),
				vector=embed(c.text),
				payload={"text": c.text, "url": url}
			)
			points.append(row)

	print(points)

	#создание индекса
	#TODO:проверка существования
	QUADRANT_CLIENT.create_collection(
		collection_name="itmo",
		vectors_config=VectorParams(size=EMBED_SIZE, distance=Distance.COSINE),
	)

	QUADRANT_CLIENT.upsert(
		collection_name="itmo",
		points=points
	)
    
    
    

