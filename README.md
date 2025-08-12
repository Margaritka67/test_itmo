# AI-ассистент для ответов на вопросы о программах высшего образования ИТМО
В проекте использован RAG-пайплаин.

Для парсинга данных с сайта используется unstructured.
Для хранения данных база qdrant

База разворачивается дополнительно через докер
docker run -p 6333:6333 qdrant/qdrant

Если хотим переиспользовать индексы, docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

Для создания индекса и обращения к ллм используются langchain компоненты
ЛЛМ и Эмбединги могут быть любые OpenAI-совместимые


енвы для запуска:

QDRANT_URL=http://localhost:6333/

CHAT_API_KEY='tgp_v1_...'
CHAT_MODEL='Qwen/Qwen3-235B-A22B-Instruct-2507-tput'
CHAT_API_BASE='https://api.together.xyz/v1/'

EMBED_MODEL="intfloat/multilingual-e5-large-instruct"
EMBED_API_BASE='https://api.together.xyz/v1/'
EMBED_API_KEY='tgp_v1_...'
EMBED_SIZE=1024
