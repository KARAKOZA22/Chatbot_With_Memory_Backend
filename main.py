import os
import json
import uuid
import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel
import time
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import nest_asyncio
from os import environ as env


log_path = "./chatbot_memory.log"

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class NotebookMemoryStore:
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        logging.info("🚀 Initializing NotebookMemoryStore...")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = "chatbot_with_memory"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        # Simple limits
        self.max_total_messages = 30   # Total messages to keep
        self.max_summaries =  5        # Max summaries before recursive compression
        self.max_regular = 25           # How many messages to summarize at once

        logging.info(f"📊 Configuration: max_messages={self.max_total_messages}, max_summaries={self.max_summaries}, max_regular={self.max_regular}")

        # Initialize collection
        self._ensure_collection()

        logging.info(f"✅ NotebookMemoryStore initialized! Embedding size: {self.embedding_size}")

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            logging.info(f"🔍 Checking if collection '{self.collection_name}' exists...")
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                logging.info(f"📦 Creating new collection '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_size,
                        distance=models.Distance.COSINE
                    )
                )
                logging.info(f"✅ Collection '{self.collection_name}' created successfully!")
            else:
                logging.info(f"✅ Collection '{self.collection_name}' already exists")
        except Exception as e:
            logging.error(f"Failed to ensure collection: {str(e)}")
            raise

    def store_message(self, message: str, message_type: str = "user"):
        logging.info(f"💾 Storing {message_type} message (length: {len(message)} chars)")
        logging.debug(f"Message content: {message[:100]}...")

        try:
            # Generate embedding
            logging.debug("🧮 Generating embedding...")
            embedding = self.embedding_model.encode(message).tolist()

            message_id = str(uuid.uuid4())
            payload = {
                "message": message,
                "message_type": message_type,
                "timestamp": datetime.now().isoformat(),
                "message_id": message_id
            }

            point = models.PointStruct(
                id=message_id,
                vector=embedding,
                payload=payload
            )

            logging.debug(f"📤 Upserting point with ID: {message_id}")
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[point]
            )

            logging.info(f"✅ {message_type.capitalize()} message stored successfully!")
            return True

        except Exception as e:
            logging.error(f"Failed to store {message_type} message: {str(e)}")
            return False

    def get_message_count(self) -> int:
        try:
            logging.debug("📊 Getting message count...")
            response = self.client.count(collection_name=self.collection_name)
            count = response.count
            logging.debug(f"Current message count: {count}")
            return count
        except Exception as e:
            logging.error(f"Failed to get message count: {str(e)}")
            return 0

    def get_all_messages_sorted(self):
        """Get all messages sorted by timestamp"""
        try:
            logging.debug("📥 Fetching all messages...")
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            messages = []
            for r in results:
                if hasattr(r, 'payload') and r.payload:
                    if "message" in r.payload and "timestamp" in r.payload:
                        messages.append(r.payload)

            messages.sort(key=lambda x: x["timestamp"])
            logging.debug(f"Retrieved and sorted {len(messages)} messages")
            return messages

        except Exception as e:
            logging.error(f"Failed to fetch messages: {str(e)}")
            return []

    def get_context_messages(self):
        """Get all messages for LLM context (will be max 100)"""
        try:
            logging.info("🔄 Building context messages for LLM...")
            all_messages = self.get_all_messages_sorted()

            # Convert to LLM format
            context_messages = []
            message_type_counts = {"user": 0, "assistant": 0, "summary": 0}

            for msg in all_messages:
                if msg['message_type'] == 'user':
                    context_messages.append({"role": "user", "content": msg['message']})
                    message_type_counts["user"] += 1
                elif msg['message_type'] == 'assistant':
                    context_messages.append({"role": "assistant", "content": msg['message']})
                    message_type_counts["assistant"] += 1
                elif msg['message_type'] == 'summary':
                    # Include summaries as system context
                    context_messages.append({"role": "system", "content": f"Previous context: {msg['message']}"})
                    message_type_counts["summary"] += 1

            logging.info(f"📋 Context built: {len(context_messages)} total messages")
            logging.info(f"   └─ User: {message_type_counts['user']}, Assistant: {message_type_counts['assistant']}, Summaries: {message_type_counts['summary']}")

            return context_messages

        except Exception as e:
            logging.error(f"Failed to build context: {str(e)}")
            return []

    def cleanup_memory(self):
        """Clean up memory when we exceed 100 total messages"""
        try:
            message_count = self.get_message_count()
            logging.info(f"🧹 Memory cleanup check: {message_count}/{self.max_total_messages} messages")

            if message_count <= self.max_total_messages:
                logging.info("✅ No cleanup needed - under message limit")
                return  # No cleanup needed

            logging.warning(f"⚠️  Memory cleanup required! Current: {message_count}, Max: {self.max_total_messages}")

            all_messages = self.get_all_messages_sorted()

            # Separate message types
            regular_messages = [m for m in all_messages if m['message_type'] in ['user', 'assistant']]
            summaries = [m for m in all_messages if m['message_type'] == 'summary']

            logging.info(f"📊 Message breakdown: {len(regular_messages)} regular, {len(summaries)} summaries")

            # Step 1: Handle regular message overflow
            if len(regular_messages) >= self.max_regular:
                logging.info("🔄 Step 1: Processing regular message overflow...")
                """
                # Calculate how many regular messages to remove
                target_regular_count = self.max_total_messages - len(summaries) - self.batch_size
                if target_regular_count < 20:  # Keep at least 20 recent messages
                    target_regular_count = 20
                messages_to_remove = regular_messages[:len(regular_messages) - target_regular_count]"""

                if regular_messages:
                    logging.info(f"📝 Creating summary for {len(regular_messages)} messages...")

                    # Create summary of removed messages
                    summary_text = chat_assistant.create_summary_from_messages(regular_messages)
                    logging.info(f"✅ Summary created (length: {len(summary_text)} chars)")

                    # Delete old messages
                    ids_to_delete = [m['message_id'] for m in regular_messages]
                    logging.info(f"🗑️  Deleting {len(ids_to_delete)} old messages...")
                    self.delete_messages_by_ids(ids_to_delete)

                    # Store new summary
                    logging.info("💾 Storing new summary...")
                    self.store_message(summary_text, message_type="summary")
                    logging.info(f"✅ Step 1 complete: Summarized and removed {len(regular_messages)} regular messages")

            # Step 2: Handle summary overflow (recursive summarization)
            updated_summaries = self.get_summaries()  # Get fresh list after step 1

            if len(updated_summaries) > self.max_summaries:
                logging.warning(f"🔄 Step 2: Too many summaries ({len(updated_summaries)}/{self.max_summaries}), creating meta-summary...")

                # Create meta-summary
                meta_summary_text = chat_assistant.create_meta_summary_from_summaries(updated_summaries)
                logging.info(f"✅ Meta-summary created (length: {len(meta_summary_text)} chars)")

                # Delete old summaries
                summary_ids_to_delete = [s['message_id'] for s in updated_summaries]
                logging.info(f"🗑️  Deleting {len(summary_ids_to_delete)} compressed summaries...")
                self.delete_messages_by_ids(summary_ids_to_delete)

                # Store meta-summary
                logging.info("💾 Storing meta-summary...")
                self.store_message(meta_summary_text, message_type="summary")
                logging.info(f"✅ Step 2 complete: Created meta-summary from {len(updated_summaries)} summaries")

            final_count = self.get_message_count()
            logging.info(f"🎉 Cleanup complete! Messages: {message_count} → {final_count}")

        except Exception as e:
            logging.error(f"Memory cleanup failed: {str(e)}")

    def get_summaries(self):
        """Get all summary messages"""
        try:
            logging.debug("📚 Fetching all summaries...")
            all_messages = self.get_all_messages_sorted()
            summaries = [m for m in all_messages if m['message_type'] == 'summary']
            logging.debug(f"Found {len(summaries)} summaries")
            return summaries
        except Exception as e:
            logging.error(f"Failed to get summaries: {str(e)}")
            return []

    def delete_messages_by_ids(self, message_ids: List[str]):
        try:
            logging.debug(f"🗑️  Deleting {len(message_ids)} messages by ID...")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=message_ids),
                wait = True
            )
            time.sleep(1)
            logging.info(f"✅ Successfully deleted {len(message_ids)} messages")
        except Exception as e:
            logging.error(f"Failed to delete messages: {str(e)}")


class NotebookChatAssistant:
    def __init__(self, groq_api: str, chutes_api: str, memory_store: NotebookMemoryStore):
        logging.info("🤖 Initializing NotebookChatAssistant...")
        nest_asyncio.apply()

        # Initialize both API clients
        self.groq_api_key = groq_api
        self.chutes_api_key = chutes_api
        self.groq_client = Groq(api_key=groq_api)
        self.groq_model = "llama-3.3-70b-versatile"
        self.chutes_model = "chutesai/Mistral-Small-3.1-24B-Instruct-2503"
        self.memory_store = memory_store

        logging.info(f"🎯 Primary Model (Groq): {self.groq_model}")
        logging.info(f"🎯 Fallback Model (Chutes): {self.chutes_model}")
        logging.info("✅ NotebookChatAssistant initialized!")

    def _try_groq_response(self, messages: List[Dict[str, str]]) -> str:
        """Try to generate response using Groq API"""
        try:
            logging.info("🚀 Attempting Groq API request...")

            completion = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=1,
                max_completion_tokens=1024,
            )

            response = completion.choices[0].message.content
            logging.info(f"✅ Groq response successful! Length: {len(response)} characters")
            return response

        except Exception as e:
            logging.warning(f"⚠️ Groq API failed: {str(e)}")
            return None

    async def _stream_response(self, messages: List[Dict[str, str]]) -> str:
        """Fallback to Chutes API with streaming"""
        logging.info(f"🌐 Starting Chutes streaming request with {len(messages)} messages")

        headers = {
            "Authorization": f"Bearer {self.chutes_api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model": self.chutes_model,
            "messages": messages,
            "stream": True,
            "max_tokens": 1024,
            "temperature": 1
        }

        response_text = ""
        chunk_count = 0

        try:
            logging.debug("📡 Creating HTTP session...")
            async with aiohttp.ClientSession() as session:
                logging.debug("📤 Sending POST request to Chutes API...")
                async with session.post(
                    "https://llm.chutes.ai/v1/chat/completions",
                    headers=headers,
                    json=body
                ) as response:
                    logging.info(f"📡 Response status: {response.status}")

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                logging.debug("🏁 Stream completed")
                                break
                            try:
                                json_data = json.loads(data)
                                choices = json_data.get("choices")
                                if choices and len(choices) > 0:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        response_text += content
                                        chunk_count += 1
                                        if chunk_count % 10 == 0:
                                            logging.debug(f"📝 Received {chunk_count} chunks, current length: {len(response_text)}")
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logging.warning(f"Error parsing chunk: {str(e)}")
        except Exception as e:
            logging.error(f"Chutes stream response failed: {str(e)}")
            return "Sorry, I encountered an error while generating the response."

        logging.info(f"✅ Chutes stream complete! Generated {len(response_text)} characters from {chunk_count} chunks")
        return response_text

    def generate_response(self, message: str) -> str:
        """Generate response using Groq first, then Chutes as fallback"""
        logging.info(f"🎭 Generating response for message: '{message[:50]}...'")

        try:
            # Get all context messages
            logging.info("📋 Retrieving context messages...")
            context_messages = self.memory_store.get_context_messages()

            # Build full conversation
            messages = [{"role": "system", "content": "You are a helpful assistant with access to conversation history."}]
            messages.extend(context_messages)
            messages.append({"role": "user", "content": message})

            logging.info(f"💬 Total conversation length: {len(messages)} messages")

            # Try Groq first
            response = self._try_groq_response(messages)

            if response is not None:
                logging.info("✅ Response generated successfully via Groq!")
                return response

            # Fallback to Chutes
            logging.info("🔄 Falling back to Chutes API...")
            response = asyncio.run(self._stream_response(messages))

            logging.info(f"✅ Response generated successfully via Chutes! Length: {len(response)} characters")
            return response

        except Exception as e:
            logging.error(f"Response generation failed: {str(e)}")
            return "Sorry, I had trouble generating a response."

    def create_summary_from_messages(self, messages: List[Dict]) -> str:
        """Create summary from a list of messages using Groq first, then Chutes"""
        logging.info(f"📝 Creating summary from {len(messages)} messages...")

        try:
            formatted = "\n".join([f"{m['message_type'].capitalize()}: {m['message']}" for m in messages])
            logging.debug(f"Formatted message length: {len(formatted)} characters")

            system_prompt = """Summarize the following conversation segment concisely.
            Focus on key information, decisions, and context that should be remembered."""

            summary_input = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted}
            ]

            # Try Groq first
            logging.info("🤖 Attempting summary generation via Groq...")
            summary = self._try_groq_response(summary_input)

            if summary is not None:
                logging.info("✅ Summary created via Groq!")
                return summary

            # Fallback to Chutes
            logging.info("🔄 Falling back to Chutes for summary...")
            summary = asyncio.run(self._stream_response(summary_input))
            logging.info(f"✅ Summary created via Chutes! Length: {len(summary)} characters")

            return summary

        except Exception as e:
            logging.error(f"Summary creation failed: {str(e)}")
            return "Summary creation failed."

    def create_meta_summary_from_summaries(self, summaries: List[Dict]) -> str:
        """Create compressed meta-summary from multiple summaries using Groq first, then Chutes"""
        logging.info(f"📚 Creating meta-summary from {len(summaries)} summaries...")

        try:
            formatted = "\n".join([f"Previous summary: {s['message']}" for s in summaries])
            logging.debug(f"Formatted summaries length: {len(formatted)} characters")

            system_prompt = """Create a compressed master summary from these previous summaries.
            Extract and combine the most important information, themes, and context."""

            summary_input = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted}
            ]

            # Try Groq first
            logging.info("🤖 Attempting meta-summary generation via Groq...")
            meta_summary = self._try_groq_response(summary_input)

            if meta_summary is not None:
                logging.info("✅ Meta-summary created via Groq!")
                return meta_summary

            # Fallback to Chutes
            logging.info("🔄 Falling back to Chutes for meta-summary...")
            meta_summary = asyncio.run(self._stream_response(summary_input))
            logging.info(f"✅ Meta-summary created via Chutes! Length: {len(meta_summary)} characters")

            return meta_summary

        except Exception as e:
            logging.error(f"Meta-summary creation failed: {str(e)}")
            return "Meta-summary creation failed."


# Update the memory store to use the chat assistant for summarization
def update_memory_store_with_chat_assistant(memory_store, chat_assistant):
    """Update memory store methods to use chat assistant for summarization"""
    logging.info("🔗 Connecting memory store with chat assistant for summarization...")

    def create_summary(messages: List[Dict]) -> str:
        return chat_assistant.create_summary_from_messages(messages)

    def create_meta_summary(summaries: List[Dict]) -> str:
        return chat_assistant.create_meta_summary_from_summaries(summaries)

    # Replace placeholder methods
    memory_store.create_summary = create_summary
    memory_store.create_meta_summary = create_meta_summary
    logging.info("✅ Memory store updated with chat assistant for summarization!")

# ---------- FastAPI Setup ----------
logging.info("🚀 Setting up FastAPI application...")

app = FastAPI(title="Chatbot with Memory", version="1.0.0")

# Initialize components
logging.info("🔧 Loading environment variables...")
QDRANT_URL = env["QDRANT_URL"]
QDRANT_API_KEY = env["QDRANT_API_KEY"]
CHUTES_API_KEY = env["CHUTES_API_KEY"]
GROQ_API_KEY = env['GROQ_API_KEY']

# Validate environment variables
missing_vars = []
if not QDRANT_URL:
    missing_vars.append("QDRANT_URL")
if not QDRANT_API_KEY:
    missing_vars.append("QDRANT_API_KEY")
if not CHUTES_API_KEY:
    missing_vars.append("CHUTES_API_KEY")

if missing_vars:
    print(missing_vars)
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logging.error(error_msg)
    raise ValueError(error_msg)

logging.info("✅ All environment variables loaded")

# Initialize memory store and chat assistant
logging.info("🏗️  Initializing components...")
memory_store = NotebookMemoryStore(qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_API_KEY)
chat_assistant = NotebookChatAssistant(groq_api=GROQ_API_KEY, chutes_api=CHUTES_API_KEY, memory_store=memory_store)

update_memory_store_with_chat_assistant(memory_store, chat_assistant)
logging.info("🎉 All components initialized successfully!")

class ResponseRequest(BaseModel):
    message: str

@app.post("/generate-response")
def generate_response(request: ResponseRequest):
    request_id = str(uuid.uuid4())[:8]  # Short ID for tracking
    logging.info(f"🌟 [REQ-{request_id}] New request received")

    try:
        user_message = request.message
        logging.info(f"📨 [REQ-{request_id}] User message: '{user_message[:100]}...'")

        # Generate response using all available context
        logging.info(f"🤖 [REQ-{request_id}] Generating assistant response...")
        assistant_response = chat_assistant.generate_response(user_message)

        # Store both messages
        logging.info(f"💾 [REQ-{request_id}] Storing user message...")
        memory_store.store_message(user_message, message_type="user")

        logging.info(f"💾 [REQ-{request_id}] Storing assistant response...")
        memory_store.store_message(assistant_response, message_type="assistant")

        # Clean up memory if needed
        logging.info(f"🧹 [REQ-{request_id}] Running memory cleanup check...")
        memory_store.cleanup_memory()

        logging.info(f"✅ [REQ-{request_id}] Request completed successfully!")
        return {"response": assistant_response}

    except Exception as e:
        logging.error(f"❌ [REQ-{request_id}] Request failed: {str(e)}")
        return {"response": "Sorry, I encountered an error processing your request."}