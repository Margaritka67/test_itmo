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
CHAT_MODEL='Qwen/Qwen2.5-14B'
CHAT_API_BASE='https://api.together.xyz/v1/'

EMBED_MODEL="togethercomputer/m2-bert-80M-32k-retrieval"
EMBED_API_BASE='https://api.together.xyz/v1/'
EMBED_API_KEY='tgp_v1_...'