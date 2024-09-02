import json
import os
import uuid
from channels.generic.websocket import AsyncWebsocketConsumer
from openai import AsyncOpenAI, OpenAI
from .models import ChatConversation
from django.template.loader import render_to_string
from channels.db import database_sync_to_async
import faiss
import numpy as np


class ChatConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.faiss_index = None
        self.embeddings = None
        self.documents = None
        self.metadata = None
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def connect(self):
        self.user = self.scope["user"]
        self.conversation_id = self.scope['url_route']['kwargs']['conversation_id']
        if self.user.is_authenticated:
            self.messages = await self.fetch_conversation(self.conversation_id)
            await self.initialize_faiss()
            await self.accept()
        else:
            await self.close()

    @database_sync_to_async
    def initialize_faiss(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        faiss_index_path = os.path.join(base_dir, "django_chatbot_app", "documents", "output_docs", "faiss_index.faiss")
        documents_path = os.path.join(base_dir, "django_chatbot_app", "documents", "output_docs", "documents.npy")
        metadata_path = os.path.join(base_dir, "django_chatbot_app", "documents", "output_docs", "metadata.json")

        self.faiss_index = faiss.read_index(faiss_index_path)
        self.documents = np.load(documents_path, allow_pickle=True)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    async def disconnect(self, close_code):
        if self.user.is_authenticated:
            await self.save_conversation(self.conversation_id, self.messages)
            # delete conversation if no messages
            if not self.messages:
                await self.delete_conversation(self.conversation_id)

    async def retrieve_relevant_documents(self, query, k=4):
        query_vector = await self.encode_query(query)
        distances, indices = await self.search_faiss(query_vector, k)

        relevant_docs = []
        for i in indices[0]:
            doc = self.documents[i]
            meta = self.metadata[i]
            relevant_docs.append((doc, meta['source']))

        return relevant_docs

    @database_sync_to_async
    def encode_query(self, query):
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        return embedding

    @database_sync_to_async
    def search_faiss(self, query_vector, k):
        return self.faiss_index.search(np.array([query_vector]), k)

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message_text = text_data_json["message"]

        if not message_text.strip():
            return

        relevant_docs = await self.retrieve_relevant_documents(message_text)

        context = "\n\n".join([f"Document Titel: {source}\n Inhoud: {doc}" for doc, source in relevant_docs])

        print(context)

        prompt = f"""
Je bent een expert op het gebied van voedselveiligheid in de Europese Unie. 
Je bent verantwoordelijk voor het verstrekken van correcte informatie aan de Europese burgers.
Denk stap voor stap voordat je de vraag beantwoordt.

Instructies:
1. Beantwoordt alleen vragen over voedselveiligheid.
2. Beantwoordt de vraag op basis van de gegeven context.
3. Benoem altijd de titel van de gebruikte documenten in het antwoord.
4. Als er niet genoeg relevante informatie is om de vraag te beantwoorden, geef dit dan aan. Verzin geen antwoord.

Context: {context}

Vraag: {message_text}

Antwoord:"""

        self.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        user_message_html = render_to_string(
            "websocket_partials/user_message.html",
            {"message_text": message_text},
        )
        await self.send(text_data=user_message_html)

        message_id = uuid.uuid4().hex
        contents_div_id = f"message-response-{message_id}"
        system_message_html = render_to_string(
            "websocket_partials/system_message.html",
            {"contents_div_id": contents_div_id},
        )
        await self.send(text_data=system_message_html)

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        stream = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=self.messages,
            stream=True,
        )

        full_message = ""
        async for chunk in stream:
            message_chunk = chunk.choices[0].delta.content
            if message_chunk:
                full_message += message_chunk
                chunk_html = f'<div hx-swap-oob="beforeend:#{contents_div_id}">{message_chunk}</div>'
                await self.send(text_data=chunk_html)

        self.messages.append(
            {
                "role": "assistant",
                "content": full_message,
            }
        )
        final_message = render_to_string(
            "websocket_partials/final_system_message.html",
            {
                "contents_div_id": contents_div_id,
                "message": full_message,
            },
        )
        await client.close()
        await self.send(text_data=final_message)

    @database_sync_to_async
    def fetch_conversation(self, id):
        chat = ChatConversation.objects.get(id=id, user=self.user)
        return chat.conversation if chat.conversation else []

    @database_sync_to_async
    def save_conversation(self, id, new_messages):
        chat = ChatConversation.objects.get(id=id, user=self.user)
        # Remove the context from user messages before saving
        cleaned_messages = []
        for message in new_messages:
            if message["role"] == "user":
                content = message["content"].split("Vraag: ")[-1].split("\n\nAntwoord:")[0]
                cleaned_messages.append({"role": "user", "content": content})
            else:
                cleaned_messages.append(message)
        chat.conversation = cleaned_messages
        chat.save()

    @database_sync_to_async
    def delete_conversation(self, id):
        try:
            chat = ChatConversation.objects.get(id=id, user=self.user)
            if not chat.conversation:  # Double-checking in case messages were added
                chat.delete()
        except ChatConversation.DoesNotExist:
            pass
