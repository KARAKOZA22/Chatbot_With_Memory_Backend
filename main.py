import streamlit as st
import uuid
import json
import time
import logging
import warnings
from datetime import datetime
from typing import List, Dict, Union, Any

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
            st.stop() # Stop the app if Qdrant isn't ready

    def store_message(self, message: str, message_type: str = "user", chat_id: str = None):
        logging.info(f"💾 Storing {message_type} message (length: {len(message)} chars) for chat_id: {chat_id}")
        if chat_id is None:
            logging.warning("Attempted to store message without a chat_id. This message will not be associated with a session.")
            return False # Or raise an error, depending on desired strictness

        try:
            embedding = self.embedding_model.encode(message).tolist()
            message_uuid = str(uuid.uuid4()) # Use a unique ID for each message point
            payload = {
                "message": message,
                "message_type": message_type,
                "timestamp": datetime.now().isoformat(),
                "message_id": message_uuid,
                "chat_id": chat_id # Store the chat ID
            }
            point = models.PointStruct(id=message_uuid, vector=embedding, payload=payload)
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
        """Get all messages sorted by timestamp across all chats"""
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000, # Max limit, adjust if you expect more messages
                with_payload=True,
                with_vectors=False,
            )
            messages = []
            for r in results:
                if hasattr(r, 'payload') and r.payload:
                    if "message" in r.payload and "timestamp" in r.payload and "chat_id" in r.payload:
                        messages.append(r.payload)
            messages.sort(key=lambda x: x["timestamp"])
            logging.debug(f"Retrieved and sorted {len(messages)} messages from all chats.")
            return messages
        except Exception as e:
            logging.error(f"Failed to fetch all messages: {str(e)}")
            return []

    def load_all_chats(self) -> Dict[str, Dict[str, Any]]:
        """Loads all chat history from Qdrant, grouped by chat_id."""
        logging.info("⏳ Loading all chat history from Qdrant...")
        all_persisted_messages = self.get_all_messages_sorted()

        loaded_chats: Dict[str, Dict[str, Any]] = {}
        for msg_payload in all_persisted_messages:
            chat_id = msg_payload.get("chat_id")
            if not chat_id:
                logging.warning(f"Skipping message due to missing chat_id: {msg_payload.get('message_id')}")
                continue

            if chat_id not in loaded_chats:
                # Initialize chat with a default title for now, will refine later
                loaded_chats[chat_id] = {
                    "title": f"Chat {datetime.fromisoformat(msg_payload['timestamp']).strftime('%Y-%m-%d')}",
                    "messages": []
                }
            loaded_chats[chat_id]["messages"].append({
                "role": msg_payload["message_type"],
                "content": msg_payload["message"]
            })

            # Update the chat title to reflect the date of the first message
            # The list is sorted, so the first message added for a chat_id is the earliest
            if len(loaded_chats[chat_id]["messages"]) == 1:
                first_message_time = datetime.fromisoformat(msg_payload['timestamp'])
                loaded_chats[chat_id]["title"] = f"Chat {first_message_time.strftime('%Y-%m-%d')}"
            else:
                # If a more precise title is desired (e.g., using the *earliest* message from the group),
                # this logic could be enhanced. For simplicity, we just use the first message's date.
                pass # Title is already set by the first message for this chat_id

        logging.info(f"✅ Loaded {len(loaded_chats)} chat sessions from Qdrant.")
        return loaded_chats


    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get all messages for LLM context from the entire history."""
        try:
            logging.info("🔄 Building context messages for LLM from all available history (no chat_id filtering)...")
            
            # Get all messages from the database, which are already sorted by timestamp
            all_persisted_messages = self.get_all_messages_sorted()

            context_messages = []
            message_type_counts = {"user": 0, "assistant": 0, "summary": 0}

            for msg in all_persisted_messages:
                # Ensure we only include user/assistant/summary types for LLM context
                if msg['message_type'] == 'user':
                    context_messages.append({"role": "user", "content": msg['message']})
                    message_type_counts["user"] += 1
                elif msg['message_type'] == 'assistant':
                    context_messages.append({"role": "assistant", "content": msg['message']})
                    message_type_counts["assistant"] += 1
                elif msg['message_type'] == 'summary':
                    context_messages.append({"role": "system", "content": f"Previous context: {msg['message']}"})
                    message_type_counts["summary"] += 1

            logging.info(f"📋 Context built: {len(context_messages)} total messages")
            logging.info(f"   └─ User: {message_type_counts['user']}, Assistant: {message_type_counts['assistant']}, Summaries: {message_type_counts['summary']}")

            return context_messages

        except Exception as e:
            logging.error(f"Failed to build context: {str(e)}")
            return []


    def cleanup_memory(self):
        """Clean up memory when we exceed max_total_messages (this is a global cleanup)"""
        try:
            message_count = self.get_message_count()
            logging.info(f"🧹 Global Memory cleanup check: {message_count}/{self.max_total_messages} messages")

            if message_count <= self.max_total_messages:
                logging.info("✅ No global cleanup needed - under message limit")
                return

            logging.warning(f"⚠️  Global Memory cleanup required! Current: {message_count}, Max: {self.max_total_messages}")

            all_messages = self.get_all_messages_sorted() # Gets ALL messages across all chats

            # Separate message types
            regular_messages = [m for m in all_messages if m['message_type'] in ['user', 'assistant']]
            summaries = [m for m in all_messages if m['message_type'] == 'summary']

            logging.info(f"📊 Message breakdown (global): {len(regular_messages)} regular, {len(summaries)} summaries")

            # Step 1: Handle regular message overflow by summarizing oldest messages
            if len(regular_messages) > self.max_regular:
                logging.info(f"🔄 Step 1: Processing global regular message overflow. Need to summarize oldest {len(regular_messages) - self.max_regular} messages.")
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
                    self.store_message(summary_text, message_type="summary", chat_id="global_summary") # Use a special chat_id for global summaries
                    logging.info(f"✅ Step 1 complete: Summarized and removed {len(ids_to_delete)} regular messages")

            updated_summaries = self.get_summaries() # Get global summaries
            if len(updated_summaries) > self.max_summaries:
                logging.warning(f"🔄 Step 2: Too many global summaries ({len(updated_summaries)}/{self.max_summaries}), creating meta-summary...")
                if not hasattr(self, 'create_meta_summary') or not callable(self.create_meta_summary):
                    logging.error("Meta-summary function 'create_meta_summary' not linked to MemoryStore. Cannot meta-summarize.")
                    return
                meta_summary_text = self.create_meta_summary(updated_summaries)
                logging.info(f"✅ Meta-summary created (length: {len(meta_summary_text)} chars)")

                summary_ids_to_delete = [s['message_id'] for s in updated_summaries]
                logging.info(f"🗑️  Deleting {len(summary_ids_to_delete)} compressed summaries...")
                self.delete_messages_by_ids(summary_ids_to_delete)

                logging.info("💾 Storing meta-summary...")
                self.store_message(meta_summary_text, message_type="summary", chat_id="global_summary") # Use a special chat_id for global summaries
                logging.info(f"✅ Step 2 complete: Created meta-summary from {len(updated_summaries)} summaries")

            final_count = self.get_message_count()
            logging.info(f"🎉 Global cleanup complete! Messages now: {final_count}")

        except Exception as e:
            logging.error(f"Memory cleanup failed: {str(e)}")

    def get_summaries(self):
        """Get all summary messages (potentially filtered by chat_id if needed, currently global)"""
        try:
            logging.debug("📚 Fetching all summaries...")
            all_messages = self.get_all_messages_sorted() # This already returns messages with chat_id
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

    def generate_response(self, message: str) -> str: # Removed current_chat_id parameter
        logging.info(f"🎭 Generating response for message: '{message[:50]}...'")
        if not self.memory_store:
            logging.error("Memory store not set for ChatAssistant. Cannot generate response.")
            return "Error: Chatbot memory is not initialized."
        try:
            # No longer pass current_chat_id to get_context_messages
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
            Focus on key information, decisions, and context that should be remembered.
            The summary should be no longer than 150 words."""
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
            Extract and combine the most important information, themes, and context.
            The meta-summary should be no longer than 200 words."""
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
        st.stop() # Stop the app if initialization fails

# Access initialized components
memory_store = st.session_state.memory_store
chat_assistant = st.session_state.chat_assistant


# Initialize session state for chats management and load from Qdrant
if "chats" not in st.session_state:
    st.session_state.chats = memory_store.load_all_chats()
    logging.info(f"Loaded {len(st.session_state.chats)} chats from Qdrant during startup.")

    # If no chats were loaded, create a new one
    if not st.session_state.chats:
        chat_id = str(uuid.uuid4())
        title = f"Chat {datetime.now().strftime('%Y-%m-%d')}" # Initial title is just the date
        st.session_state.chats[chat_id] = {"title": title, "messages": []}
        st.session_state.current_chat = chat_id
        logging.info(f"Created new initial chat: {chat_id}")
    else:
        # If chats were loaded, select the most recent one
        # Sort by the datetime in the title or by the internal timestamp of the first message if available
        most_recent_chat_id = sorted(
            st.session_state.chats.keys(),
            key=lambda cid: datetime.strptime(st.session_state.chats[cid]["title"].replace("Chat ", ""), '%Y-%m-%d') if "Chat " in st.session_state.chats[cid]["title"] else datetime.min,
            reverse=True
        )[0]
        st.session_state.current_chat = most_recent_chat_id
        logging.info(f"Selected most recent loaded chat: {most_recent_chat_id}")
    st.rerun() # Rerun to display the loaded/selected chat


# Sidebar for chat management
with st.sidebar:
    st.header("Your Chats")

    st.markdown("---")

    if st.session_state.chats:
        # Sort chats by date for display
        sorted_chat_ids = sorted(
            st.session_state.chats.keys(),
            key=lambda cid: datetime.strptime(st.session_state.chats[cid]["title"].replace("Chat ", ""), '%Y-%m-%d') if "Chat " in st.session_state.chats[cid]["title"] else datetime.min,
            reverse=True
        )
        for chat_id in sorted_chat_ids:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(
                    st.session_state.chats[chat_id]["title"],
                    key=f"select_{chat_id}",
                    use_container_width=True,
                    type="primary" if chat_id == st.session_state.current_chat else "secondary"
                ):
                    st.session_state.current_chat = chat_id
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{chat_id}"):
                    # Delete messages associated with this chat_id from Qdrant
                    messages_to_delete_qdrant_ids = [
                        msg_payload["message_id"] for msg_payload in memory_store.get_all_messages_sorted()
                        if msg_payload.get("chat_id") == chat_id
                    ]
                    if messages_to_delete_qdrant_ids:
                        memory_store.delete_messages_by_ids(messages_to_delete_qdrant_ids)
                        logging.info(f"Deleted {len(messages_to_delete_qdrant_ids)} messages from Qdrant for chat {chat_id}.")

                    # Remove chat from session state
                    if chat_id == st.session_state.current_chat:
                        st.session_state.current_chat = None
                    del st.session_state.chats[chat_id]
                    logging.info(f"Deleted chat {chat_id} from session state.")
                    st.rerun()
    else:
        st.write("No chats available. Start typing to create a new one automatically!")


# Main chat display area
if st.session_state.current_chat:
    current_chat = st.session_state.chats[st.session_state.current_chat]
    st.header(current_chat["title"])

    # Reload messages for the current chat from Qdrant to ensure consistency
    # This is important after cleanup or if other changes happened to the store
    # This now gets ALL messages for the context, not just the current chat's
    persisted_messages_for_current_chat = memory_store.get_context_messages()
    
    # Update current_chat["messages"] to reflect the persisted state
    # This part needs to be careful if we truly want to show only messages related to the selected UI chat_id.
    # The get_context_messages is now global, but the UI might still want to show a single chat.
    # To correctly display *only* the current chat's messages in the UI, even if LLM context is global,
    # we should filter `persisted_messages_for_current_chat` by `st.session_state.current_chat`
    # for display purposes only.
    # However, the user's initial request was "get all the data that is stored in the data base thats it dont filter by the key or chat id cuz only one user will be reacting with the chatbot"
    # This implies that perhaps the concept of separate "chats" in the UI is less important,
    # or that the user wants the display to also reflect the combined history.
    # For now, I'll update current_chat["messages"] with only the messages
    # that belong to the `st.session_state.current_chat` to maintain the multi-chat UI feature
    # while providing global context to the LLM. This is a common pattern.

    # Filter messages for display in the current UI chat window
    displayed_messages = [
        msg_data for msg_data in persisted_messages_for_current_chat
        if msg_data.get("chat_id") == st.session_state.current_chat # Filter for UI display
    ]

    current_chat["messages"] = []
    for msg_data in displayed_messages:
        if msg_data["role"] in ["user", "assistant"]:
            current_chat["messages"].append({
                "role": msg_data["role"],
                "content": msg_data["content"]
            })

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
                    # Pass current_chat_id to generate_response
                    assistant_reply = chat_assistant.generate_response(prompt) # Removed current_chat_id from arguments

                    # Store messages in Qdrant via memory_store, including the current_chat_id
                    memory_store.store_message(prompt, message_type="user", chat_id=st.session_state.current_chat)
                    memory_store.store_message(assistant_reply, message_type="assistant", chat_id=st.session_state.current_chat)

                    # Perform global memory cleanup
                    memory_store.cleanup_memory()

                    st.markdown(assistant_reply)
                except Exception as e:
                    st.error(f"An error occurred during response generation: {e}")
                    assistant_reply = "Sorry, I encountered an error while processing your request."
                    st.markdown(assistant_reply)

        current_chat["messages"].append({"role": "assistant", "content": assistant_reply})
        st.rerun() # Rerun to display the newly added messages
else:
    st.info("A chat session is being set up automatically, or you can select an existing one from the sidebar if available.")
