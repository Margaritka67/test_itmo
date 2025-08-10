from unstructured.partition.html import partition_html
from transformers import AutoTokenizer
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, AsyncGenerator
from langchain_openai import ChatOpenAI
from qdrant_client import AsyncQdrantClient
import os

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

CHAT_API_KEY = os.getenv("CHAT_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
CHAT_API_BASE = os.getenv("CHAT_API_BASE")

QDRANT_URL = os.getenv("QDRANT_URL")

EMBED_MODEL = os.getenv("EMBED_MODEL")
EMBED_API_BASE = os.getenv("EMBED_API_BASE")
EMBED_API_KEY = os.getenv("EMBED_API_KEY")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))
    
    
LLM_CLIENT = ChatOpenAI(
    model_name=CHAT_MODEL,
    openai_api_key=CHAT_API_KEY,
    openai_api_base=CHAT_API_BASE,
    temperature=0.05,
)

QUADRANT_CLIENT = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)

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


QUADRANT_CLIENT = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)


async def embed(text: List[str]) -> List[List[float]]:
    if len(text) == 0:
        return []
    res = await embed_client.embeddings.create(
        input=text,
        model=EMBED_MODEL,
        encoding_format="float",
    )
    return [item.embedding for item in res.data]


async def main():

	#Парсинг данных
	urls = ["https://abit.itmo.ru/program/master/ai", "https://abit.itmo.ru/program/master/ai_product"]

	for url in urls:
	    elements = partition_html(url=url)
	    print("\n\n".join([str(el) for el in elements]))
    
    
    
    
