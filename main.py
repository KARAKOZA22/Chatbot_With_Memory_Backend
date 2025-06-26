import streamlit as st
import uuid
import json
import time
import logging
import warnings
from datetime import datetime
from typing import List, Dict, Union

# --- Suppress warnings from sentence-transformers for cleaner logs during startup ---
warnings.filterwarnings("ignore", category=UserWarning, module='sentence_transformers')

# --- Libraries for backend logic (now integrated) ---
try:
    import asyncio
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    import aiohttp
    import nest_asyncio
except ImportError as e:
    st.error(f"Missing required library: {e.name}. Please install it using `pip install -r requirements.txt`.")
    st.stop() # Stop the app if crucial libraries are missing

# Apply nest_asyncio to allow asyncio.run in sync contexts (Streamlit's main loop)
nest_asyncio.apply()

# --- Logging Setup (for Streamlit's internal logs or console) ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Optionally, suppress noisy logs from underlying libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

st.set_page_config(page_title="Chatbot That Can Remember", layout="wide")
st.title("Chatbot That Remembers")

# --- NotebookMemoryStore Class (Backend Logic) ---
class NotebookMemoryStore:
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        logging.info("🚀 Initializing NotebookMemoryStore...")
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = "chatbot_with_memory"
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model: {e}")
            raise RuntimeError("Could not load embedding model. Check your internet connection or model path.")

        self.max_total_messages = 30
        self.max_summaries = 5
        self.max_regular = 25

        logging.info(f"📊 Configuration: max_messages={self.max_total_messages}, max_summaries={self.max_summaries}, max_regular={self.max_regular}")
        self._ensure_collection()
        logging.info(f"✅ NotebookMemoryStore initialized! Embedding size: {self.embedding_size}")

    def _ensure_collection(self):
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
            st.error(f"Failed to connect to Qdrant or create collection: {str(e)}. Please check your Qdrant URL and API key.")
            st.stop()

    def store_message(self, message: str, message_type: str = "user"):
        logging.info(f"💾 Storing {message_type} message (length: {len(message)} chars)")
        try:
            embedding = self.embedding_model.encode(message).tolist()
            message_id = str(uuid.uuid4())
            payload = {
                "message": message,
                "message_type": message_type,
                "timestamp": datetime.now().isoformat(),
                "message_id": message_id
            }
            point = models.PointStruct(id=message_id, vector=embedding, payload=payload)
            self.client.upsert(collection_name=self.collection_name, wait=True, points=[point])
            logging.info(f"✅ {message_type.capitalize()} message stored successfully!")
            return True
        except Exception as e:
            logging.error(f"Failed to store {message_type} message: {str(e)}")
            return False

    def get_message_count(self) -> int:
        try:
            response = self.client.count(collection_name=self.collection_name)
            return response.count
        except Exception as e:
            logging.error(f"Failed to get message count: {str(e)}")
            return 0

    def get_all_messages_sorted(self):
        try:
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
        try:
            logging.info("🔄 Building context messages for LLM...")
            all_messages = self.get_all_messages_sorted()
            context_messages = []
            for msg in all_messages:
                if msg['message_type'] == 'user':
                    context_messages.append({"role": "user", "content": msg['message']})
                elif msg['message_type'] == 'assistant':
                    context_messages.append({"role": "assistant", "content": msg['message']})
                elif msg['message_type'] == 'summary':
                    context_messages.append({"role": "system", "content": f"Previous context: {msg['message']}"})
            logging.info(f"📋 Context built: {len(context_messages)} total messages")
            return context_messages
        except Exception as e:
            logging.error(f"Failed to build context: {str(e)}")
            return []

    def cleanup_memory(self):
        try:
            message_count = self.get_message_count()
            logging.info(f"🧹 Memory cleanup check: {message_count}/{self.max_total_messages} messages")

            if message_count <= self.max_total_messages:
                logging.info("✅ No cleanup needed - under message limit")
                return

            logging.warning(f"⚠️  Memory cleanup required! Current: {message_count}, Max: {self.max_total_messages}")
            all_messages = self.get_all_messages_sorted()
            regular_messages = [m for m in all_messages if m['message_type'] in ['user', 'assistant']]
            summaries = [m for m in all_messages if m['message_type'] == 'summary']

            logging.info(f"📊 Message breakdown: {len(regular_messages)} regular, {len(summaries)} summaries")

            if len(regular_messages) > self.max_regular:
                logging.info(f"🔄 Step 1: Processing regular message overflow. Need to summarize oldest {len(regular_messages) - self.max_regular} messages.")
                messages_to_summarize_count = len(regular_messages) - self.max_regular
                messages_to_process = regular_messages[:messages_to_summarize_count]

                if messages_to_process:
                    logging.info(f"📝 Creating summary for {len(messages_to_process)} oldest regular messages...")
                    if not hasattr(self, 'create_summary') or not callable(self.create_summary):
                        logging.error("Summary function 'create_summary' not linked to MemoryStore. Cannot summarize.")
                        return
                    summary_text = self.create_summary(messages_to_process)
                    logging.info(f"✅ Summary created (length: {len(summary_text)} chars)")

                    ids_to_delete = [m['message_id'] for m in messages_to_process]
                    logging.info(f"🗑️  Deleting {len(ids_to_delete)} old regular messages...")
                    self.delete_messages_by_ids(ids_to_delete)

                    logging.info("💾 Storing new summary...")
                    self.store_message(summary_text, message_type="summary")
                    logging.info(f"✅ Step 1 complete: Summarized and removed {len(ids_to_delete)} regular messages")

            updated_summaries = self.get_summaries()
            if len(updated_summaries) > self.max_summaries:
                logging.warning(f"🔄 Step 2: Too many summaries ({len(updated_summaries)}/{self.max_summaries}), creating meta-summary...")
                if not hasattr(self, 'create_meta_summary') or not callable(self.create_meta_summary):
                    logging.error("Meta-summary function 'create_meta_summary' not linked to MemoryStore. Cannot meta-summarize.")
                    return
                meta_summary_text = self.create_meta_summary(updated_summaries)
                logging.info(f"✅ Meta-summary created (length: {len(meta_summary_text)} chars)")

                summary_ids_to_delete = [s['message_id'] for s in updated_summaries]
                logging.info(f"🗑️  Deleting {len(summary_ids_to_delete)} compressed summaries...")
                self.delete_messages_by_ids(summary_ids_to_delete)

                logging.info("💾 Storing meta-summary...")
                self.store_message(meta_summary_text, message_type="summary")
                logging.info(f"✅ Step 2 complete: Created meta-summary from {len(updated_summaries)} summaries")

            final_count = self.get_message_count()
            logging.info(f"🎉 Cleanup complete! Messages now: {final_count}")

        except Exception as e:
            logging.error(f"Memory cleanup failed: {str(e)}")

    def get_summaries(self):
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
        if not message_ids:
            logging.info("No message IDs to delete.")
            return
        try:
            logging.debug(f"🗑️  Deleting {len(message_ids)} messages by ID...")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=message_ids),
                wait=True
            )
            time.sleep(0.5)
            logging.info(f"✅ Successfully deleted {len(message_ids)} messages")
        except Exception as e:
            logging.error(f"Failed to delete messages: {str(e)}")
    
    def clear_all_history(self):
        """Deletes all points from the collection to clear the history."""
        try:
            logging.warning(f"🗑️  CLEARING ALL HISTORY from collection '{self.collection_name}'...")
            all_messages = self.get_all_messages_sorted()
            if not all_messages:
                logging.info("✅ History is already empty. No action taken.")
                return
            
            ids_to_delete = [msg['message_id'] for msg in all_messages]
            self.delete_messages_by_ids(ids_to_delete)
            logging.info("✅✅ ALL CHAT HISTORY HAS BEEN CLEARED from Qdrant.")
        except Exception as e:
            logging.error(f"Failed to clear all history: {str(e)}")
            st.error("Could not clear chat history.")


# --- NotebookChatAssistant Class (Backend Logic) ---
class NotebookChatAssistant:
    def __init__(self, groq_api: str, chutes_api: str):
        logging.info("🤖 Initializing NotebookChatAssistant...")
        self.groq_api_key = groq_api
        self.chutes_api_key = chutes_api
        self.groq_client = Groq(api_key=groq_api)
        self.groq_model = "llama-3.1-70b-versatile"
        self.chutes_model = "chutesai/Mistral-Small-3.1-24B-Instruct-2503"
        self.memory_store: Union[NotebookMemoryStore, None] = None

        logging.info(f"🎯 Primary Model (Groq): {self.groq_model}")
        logging.info(f"🎯 Fallback Model (Chutes): {self.chutes_model}")
        logging.info("✅ NotebookChatAssistant initialized!")

    def set_memory_store(self, memory_store_instance: NotebookMemoryStore):
        self.memory_store = memory_store_instance
        logging.info("🔗 Memory store linked to ChatAssistant.")

    def _try_groq_response(self, messages: List[Dict[str, str]]) -> Union[str, None]:
        try:
            logging.info("🚀 Attempting Groq API request...")
            completion = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            response = completion.choices[0].message.content
            logging.info(f"✅ Groq response successful! Length: {len(response)} characters")
            return response
        except Exception as e:
            logging.warning(f"⚠️ Groq API failed: {str(e)}")
            return None

    async def _stream_response(self, messages: List[Dict[str, str]]) -> str:
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
            "temperature": 0.7
        }
        response_text = ""
        chunk_count = 0
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://llm.chutes.ai/v1/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=300
                ) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
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
                            except json.JSONDecodeError:
                                logging.warning(f"JSONDecodeError: Could not parse data chunk: {data}")
                                continue
        except aiohttp.ClientError as e:
            logging.error(f"Chutes HTTP client error: {str(e)}")
            return "Sorry, I encountered a network error while generating the response (Chutes API)."
        except Exception as e:
            logging.error(f"Chutes stream response failed: {str(e)}")
            return "Sorry, I encountered an unexpected error while generating the response (Chutes API)."
        logging.info(f"✅ Chutes stream complete! Generated {len(response_text)} characters from {chunk_count} chunks")
        return response_text

    def generate_response(self, message: str) -> str:
        logging.info(f"🎭 Generating response for message: '{message[:50]}...'")
        if not self.memory_store:
            logging.error("Memory store not set for ChatAssistant. Cannot generate response.")
            return "Error: Chatbot memory is not initialized."
        try:
            context_messages = self.memory_store.get_context_messages()
            messages = [{"role": "system", "content": "You are a helpful assistant with access to conversation history. Keep your responses concise and to the point."}]
            messages.extend(context_messages)
            messages.append({"role": "user", "content": message})
            logging.info(f"💬 Total conversation length: {len(messages)} messages")

            response = self._try_groq_response(messages)
            if response is not None:
                logging.info("✅ Response generated successfully via Groq!")
                return response

            logging.info("🔄 Falling back to Chutes API...")
            response = asyncio.run(self._stream_response(messages))
            logging.info(f"✅ Response generated successfully via Chutes! Length: {len(response)} characters")
            return response
        except Exception as e:
            logging.error(f"Response generation failed: {str(e)}")
            return "Sorry, I had trouble generating a response."

    def create_summary_from_messages(self, messages: List[Dict]) -> str:
        logging.info(f"📝 Creating summary from {len(messages)} messages...")
        try:
            formatted = "\n".join([f"{m['message_type'].capitalize()}: {m['message']}" for m in messages])
            system_prompt = """Summarize the following conversation segment concisely.
            Focus on key information, decisions, and context that should be remembered."""
            summary_input = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted}
            ]
            summary = self._try_groq_response(summary_input)
            if summary is not None:
                logging.info("✅ Summary created via Groq!")
                return summary
            logging.info("🔄 Falling back to Chutes for summary...")
            summary = asyncio.run(self._stream_response(summary_input))
            logging.info(f"✅ Summary created via Chutes! Length: {len(summary)} characters")
            return summary
        except Exception as e:
            logging.error(f"Summary creation failed: {str(e)}")
            return "Summary creation failed."

    def create_meta_summary_from_summaries(self, summaries: List[Dict]) -> str:
        logging.info(f"📚 Creating meta-summary from {len(summaries)} summaries...")
        try:
            formatted = "\n".join([f"Previous summary: {s['message']}" for s in summaries])
            system_prompt = """Create a compressed master summary from these previous summaries.
            Extract and combine the most important information, themes, and context."""
            summary_input = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted}
            ]
            meta_summary = self._try_groq_response(summary_input)
            if meta_summary is not None:
                logging.info("✅ Meta-summary created via Groq!")
                return meta_summary
            logging.info("🔄 Falling back to Chutes for meta-summary...")
            meta_summary = asyncio.run(self._stream_response(summary_input))
            logging.info(f"✅ Meta-summary created via Chutes! Length: {len(meta_summary)} characters")
            return meta_summary
        except Exception as e:
            logging.error(f"Meta-summary creation failed: {str(e)}")
            return "Meta-summary creation failed."

# --- Streamlit App Initialization ---

# Initialize components only once using st.session_state
if "memory_store" not in st.session_state:
    try:
        logging.info("🔧 Loading API keys from Streamlit secrets...")
        QDRANT_URL = st.secrets["QDRANT_URL"]
        QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
        CHUTES_API_KEY = st.secrets["CHUTES_API_KEY"]
        GROQ_API_KEY = st.secrets['GROQ_API_KEY']
        logging.info("✅ API keys loaded.")

        st.session_state.memory_store = NotebookMemoryStore(
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY
        )
        st.session_state.chat_assistant = NotebookChatAssistant(
            groq_api=GROQ_API_KEY,
            chutes_api=CHUTES_API_KEY
        )
        # Link memory store to chat assistant and vice-versa for summarization
        st.session_state.chat_assistant.set_memory_store(st.session_state.memory_store)
        st.session_state.memory_store.create_summary = st.session_state.chat_assistant.create_summary_from_messages
        st.session_state.memory_store.create_meta_summary = st.session_state.chat_assistant.create_meta_summary_from_summaries

        logging.info("🎉 Chatbot components initialized and stored in session state!")
    except Exception as e:
        st.error(f"Error initializing chatbot components. Please check your `secrets.toml` and internet connection: {e}")
        st.stop()

# Access initialized components
memory_store = st.session_state.memory_store
chat_assistant = st.session_state.chat_assistant


# Initialize session state for chats management
# This block is now designed to load history from Qdrant on first run
if "chats" not in st.session_state:
    logging.info("🚀 First run in this session. Attempting to load chat history from Qdrant.")
    st.session_state.chats = {}
    st.session_state.current_chat = None

    all_messages_from_db = memory_store.get_all_messages_sorted()

    if all_messages_from_db:
        logging.info(f"✅ Found {len(all_messages_from_db)} messages in DB. Rebuilding chat UI.")
        # Create a single chat session to hold the entire restored history.
        chat_id = "restored_chat_session"
        title = "Restored Chat History"
        st.session_state.chats[chat_id] = {"title": title, "messages": []}

        # Populate the messages for the UI, skipping system messages/summaries
        for msg in all_messages_from_db:
            role = msg.get('message_type') # 'user' or 'assistant'
            content = msg.get('message', '')
            if role in ['user', 'assistant']:
                st.session_state.chats[chat_id]["messages"].append({"role": role, "content": content})
        
        st.session_state.current_chat = chat_id
        logging.info(f"✅ Restored {len(st.session_state.chats[chat_id]['messages'])} messages to the UI.")

    else:
        # If the DB is empty, create a new, fresh chat session.
        logging.info("✅ No messages found in DB. Creating a new chat session.")
        chat_id = str(uuid.uuid4())
        title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        st.session_state.chats[chat_id] = {"title": title, "messages": []}
        st.session_state.current_chat = chat_id

# Sidebar for chat management
with st.sidebar:
    st.header("Chat Controls")

    if st.button("New Chat", use_container_width=True):
        # Clear the backend Qdrant history
        memory_store.clear_all_history()
        
        # Reset the frontend session state
        chat_id = str(uuid.uuid4())
        title = f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        st.session_state.chats = {chat_id: {"title": title, "messages": []}}
        st.session_state.current_chat = chat_id
        st.rerun()

    st.markdown("---")
    
    # Since we now manage one persistent history, the multi-chat display is simplified.
    # If you want multi-chat, this section would need significant rework.
    if st.session_state.chats and st.session_state.current_chat:
         st.write(f"**Current Session:**")
         st.write(st.session_state.chats[st.session_state.current_chat]["title"])
    else:
        st.write("No active chat.")

# Main chat display area
if st.session_state.current_chat and st.session_state.current_chat in st.session_state.chats:
    current_chat = st.session_state.chats[st.session_state.current_chat]
    st.header(current_chat["title"])

    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your message..."):
        current_chat["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Store user message BEFORE generating response
                    memory_store.store_message(prompt, message_type="user")

                    # Generate response based on the now-updated history
                    assistant_reply = chat_assistant.generate_response(prompt)

                    # Store assistant response
                    memory_store.store_message(assistant_reply, message_type="assistant")

                    # Perform memory cleanup if necessary
                    memory_store.cleanup_memory()

                    st.markdown(assistant_reply)
                except Exception as e:
                    logging.error(f"Fatal error in chat loop: {e}")
                    st.error(f"An error occurred during response generation: {e}")
                    assistant_reply = "Sorry, I encountered an error while processing your request."
                    st.markdown(assistant_reply)

        current_chat["messages"].append({"role": "assistant", "content": assistant_reply})
        st.rerun()
else:
    st.info("Welcome! Start a conversation or click 'New Chat' to begin.")
