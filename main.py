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
st.title("My Persistent Chatbot")

# --- NotebookMemoryStore Class (Backend Logic) ---
class NotebookMemoryStore:
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        logging.info("🚀 Initializing NotebookMemoryStore...")
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = "chatbot_with_memory_single" # Using a new collection name for clarity
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
                logging.info(f"🔄 Step 1: Processing regular message overflow...")
                messages_to_summarize_count = len(regular_messages) - self.max_regular
                messages_to_process = regular_messages[:messages_to_summarize_count]

                if messages_to_process:
                    summary_text = self.create_summary(messages_to_process)
                    ids_to_delete = [m['message_id'] for m in messages_to_process]
                    self.delete_messages_by_ids(ids_to_delete)
                    self.store_message(summary_text, message_type="summary")
                    logging.info(f"✅ Step 1 complete: Summarized and removed {len(ids_to_delete)} regular messages")

            updated_summaries = self.get_summaries()
            if len(updated_summaries) > self.max_summaries:
                logging.warning(f"🔄 Step 2: Too many summaries, creating meta-summary...")
                meta_summary_text = self.create_meta_summary(updated_summaries)
                summary_ids_to_delete = [s['message_id'] for s in updated_summaries]
                self.delete_messages_by_ids(summary_ids_to_delete)
                self.store_message(meta_summary_text, message_type="summary")
                logging.info(f"✅ Step 2 complete: Created meta-summary from {len(updated_summaries)} summaries")

            final_count = self.get_message_count()
            logging.info(f"🎉 Cleanup complete! Messages now: {final_count}")

        except Exception as e:
            logging.error(f"Memory cleanup failed: {str(e)}")

    def get_summaries(self):
        try:
            all_messages = self.get_all_messages_sorted()
            return [m for m in all_messages if m['message_type'] == 'summary']
        except Exception as e:
            logging.error(f"Failed to get summaries: {str(e)}")
            return []

    def delete_messages_by_ids(self, message_ids: List[str]):
        if not message_ids: return
        try:
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
            
            # Recreate collection for a clean slate
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.embedding_size, distance=models.Distance.COSINE)
            )
            logging.info("✅✅ ALL CHAT HISTORY HAS BEEN CLEARED from Qdrant by recreating the collection.")
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
        self.groq_model = "llama-3.3-70b-versatile"
        self.chutes_model = "chutesai/Mistral-Small-3.1-24B-Instruct-2503"
        self.memory_store: Union[NotebookMemoryStore, None] = None
        logging.info("✅ NotebookChatAssistant initialized!")

    def set_memory_store(self, memory_store_instance: NotebookMemoryStore):
        self.memory_store = memory_store_instance
        logging.info("🔗 Memory store linked to ChatAssistant.")

    def _try_groq_response(self, messages: List[Dict[str, str]]) -> Union[str, None]:
        try:
            logging.info("🚀 Attempting Groq API request...")
            completion = self.groq_client.chat.completions.create(
                model=self.groq_model, messages=messages, temperature=0.7, max_tokens=1024)
            return completion.choices[0].message.content
        except Exception as e:
            logging.warning(f"⚠️ Groq API failed: {str(e)}")
            return None

    async def _stream_response(self, messages: List[Dict[str, str]]) -> str:
        logging.info(f"🌐 Starting Chutes streaming request...")
        headers = {"Authorization": f"Bearer {self.chutes_api_key}", "Content-Type": "application/json"}
        body = {"model": self.chutes_model, "messages": messages, "stream": True, "max_tokens": 1024, "temperature": 0.7}
        response_text = ""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://llm.chutes.ai/v1/chat/completions", headers=headers, json=body, timeout=300) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]": break
                            try:
                                content = json.loads(data).get("choices")[0].get("delta", {}).get("content")
                                if content: response_text += content
                            except (json.JSONDecodeError, IndexError): continue
        except Exception as e:
            logging.error(f"Chutes stream response failed: {str(e)}")
            return "Sorry, an error occurred with the fallback API."
        return response_text

    def generate_response(self, message: str) -> str:
        if not self.memory_store: return "Error: Chatbot memory is not initialized."
        try:
            context_messages = self.memory_store.get_context_messages()
            messages = [{"role": "system", "content": "You are a helpful assistant with conversation history."}]
            messages.extend(context_messages)
            messages.append({"role": "user", "content": message})
            
            response = self._try_groq_response(messages)
            if response is not None: return response

            logging.info("🔄 Falling back to Chutes API...")
            return asyncio.run(self._stream_response(messages))
        except Exception as e:
            logging.error(f"Response generation failed: {str(e)}")
            return "Sorry, I had trouble generating a response."

    def create_summary_from_messages(self, messages: List[Dict]) -> str:
        formatted = "\n".join([f"{m['message_type'].capitalize()}: {m['message']}" for m in messages])
        system_prompt = "Summarize the following conversation segment concisely. Focus on key information, decisions, and context to be remembered."
        summary_input = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted}]
        
        summary = self._try_groq_response(summary_input)
        if summary is not None: return summary
        return asyncio.run(self._stream_response(summary_input))

    def create_meta_summary_from_summaries(self, summaries: List[Dict]) -> str:
        formatted = "\n".join([f"Previous summary: {s['message']}" for s in summaries])
        system_prompt = "Create a compressed master summary from these previous summaries. Combine the most important information, themes, and context."
        summary_input = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted}]
        
        meta_summary = self._try_groq_response(summary_input)
        if meta_summary is not None: return meta_summary
        return asyncio.run(self._stream_response(summary_input))

# --- Streamlit App Initialization ---

# Initialize components only once
if "memory_store" not in st.session_state:
    try:
        QDRANT_URL = st.secrets["QDRANT_URL"]
        QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
        CHUTES_API_KEY = st.secrets["CHUTES_API_KEY"]
        GROQ_API_KEY = st.secrets['GROQ_API_KEY']
        
        st.session_state.memory_store = NotebookMemoryStore(qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_API_KEY)
        st.session_state.chat_assistant = NotebookChatAssistant(groq_api=GROQ_API_KEY, chutes_api=CHUTES_API_KEY)
        
        st.session_state.chat_assistant.set_memory_store(st.session_state.memory_store)
        st.session_state.memory_store.create_summary = st.session_state.chat_assistant.create_summary_from_messages
        st.session_state.memory_store.create_meta_summary = st.session_state.chat_assistant.create_meta_summary_from_summaries
        logging.info("🎉 Chatbot components initialized.")
    except Exception as e:
        st.error(f"Error initializing chatbot components. Check secrets and connections: {e}")
        st.stop()

memory_store = st.session_state.memory_store
chat_assistant = st.session_state.chat_assistant

# Initialize the UI message list from DB on first load
if "messages" not in st.session_state:
    logging.info("🚀 First run in session. Loading all messages from DB for UI display.")
    st.session_state.messages = []
    all_db_messages = memory_store.get_all_messages_sorted()
    for msg in all_db_messages:
        role = msg.get('message_type')
        content = msg.get('message', '')
        if role in ['user', 'assistant']:
            st.session_state.messages.append({"role": role, "content": content})
    logging.info(f"✅ UI initialized with {len(st.session_state.messages)} messages.")

# Sidebar for chat controls
with st.sidebar:
    st.header("Chat Controls")
    if st.button("🗑️ Clear Full History", use_container_width=True, type="primary"):
        # Clear the backend Qdrant history
        memory_store.clear_all_history()
        # Reset the frontend session state message list
        st.session_state.messages = []
        st.success("Chat history has been cleared!")
        time.sleep(1) # Give user time to see the message
        st.rerun()

# Main chat display area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Store user message in DB BEFORE generating response
                memory_store.store_message(prompt, message_type="user")

                # Generate response based on the now-updated history
                assistant_reply = chat_assistant.generate_response(prompt)

                # Store assistant response in DB
                memory_store.store_message(assistant_reply, message_type="assistant")

                # Perform memory cleanup if necessary
                memory_store.cleanup_memory()

                st.markdown(assistant_reply)
                # Add assistant response to UI
                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                
            except Exception as e:
                logging.error(f"Fatal error in chat loop: {e}")
                st.error(f"An error occurred: {e}")
                assistant_reply = "Sorry, an error occurred."
                # Add error message to UI to maintain conversation flow
                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # No need to rerun here as we manually update the UI within the loop
    # st.rerun() # This would cause an unnecessary double-refresh
