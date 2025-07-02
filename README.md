# CRAYON CHATBOT 🖍️

A sophisticated chatbot application built with Streamlit that maintains persistent memory across conversations using vector embeddings and intelligent memory management.

## Features

- **Persistent Memory**: Stores and retrieves conversation history using Qdrant vector database
- **Intelligent Memory Management**: Automatically summarizes old conversations to maintain context while staying within memory limits
- **Dual AI Backend**: Uses Groq API as primary service with Chutes.ai as fallback
- **Vector Embeddings**: Leverages sentence-transformers for semantic search and context retrieval
- **Clean UI**: Simple and intuitive Streamlit interface
- **Memory Cleanup**: Automatic conversation summarization when memory limits are reached

## Architecture

### Core Components

1. **NotebookMemoryStore**: Handles persistent storage and retrieval of conversations
2. **NotebookChatAssistant**: Manages AI interactions and response generation
3. **Streamlit Frontend**: Provides the user interface

### Memory Management Strategy

- **Max Total Messages**: 30 messages total
- **Max Regular Messages**: 25 user/assistant messages
- **Max Summaries**: 5 summary messages
- **Automatic Cleanup**: When limits are exceeded, older messages are summarized and compressed

## Prerequisites

### Required APIs

1. **Qdrant Cloud Account**: For vector database storage
   - Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
   - Create a cluster and get your URL and API key

2. **Groq API Key**: For primary AI responses
   - Get your API key from [Groq](https://console.groq.com/)

3. **Chutes.ai API Key**: For fallback AI responses
   - Register at [Chutes.ai](https://chutes.ai/)

### Required Python Packages

```txt
streamlit
qdrant-client
sentence-transformers
groq
aiohttp
nest-asyncio
```

## Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KARAKOZA22/Chatbot_With_Memory_Backend.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up secrets**:
   ```
   QDRANT_URL = "your-qdrant-cluster-url"
   QDRANT_API_KEY = "your-qdrant-api-key"
   GROQ_API_KEY = "your-groq-api-key"
   CHUTES_API_KEY = "your-chutes-api-key"
   ```

4. **Run the application**:
   ```bash
   streamlit run main.py
   ```

### Deployment

#### Streamlit Community Cloud

1. Push your code to GitHub
2. Connect to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Add your secrets in the deployment settings:
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `GROQ_API_KEY`
   - `CHUTES_API_KEY`

#### Other Platforms

The app can be deployed on any platform that supports Python and Streamlit:
- Heroku
- Railway
- Google Cloud Run
- AWS EC2
- DigitalOcean App Platform

## Configuration

### Memory Settings

You can adjust memory limits in the `NotebookMemoryStore` class:

```python
self.max_total_messages = 30    # Total message limit
self.max_summaries = 5          # Maximum summary messages
self.max_regular = 25           # Maximum regular messages
```

### AI Models

Current models used:
- **Groq**: `llama-3.3-70b-versatile`
- **Chutes.ai**: `chutesai/Mistral-Small-3.1-24B-Instruct-2503`

## Usage

1. **Start Conversation**: Simply type your message in the chat input
2. **View History**: All previous messages are displayed in the chat interface
3. **New Chat**: Click "New Chat" in the sidebar to clear history and start fresh
4. **Automatic Memory**: The system automatically manages memory by summarizing old conversations

## Technical Details

### Vector Embeddings

- Uses `all-MiniLM-L6-v2` model for generating embeddings
- Stores conversations with semantic search capabilities
- Enables context-aware responses based on conversation history

### Memory Management Process

1. **Storage**: Each message is embedded and stored in Qdrant
2. **Retrieval**: Context is built from stored messages for AI responses
3. **Cleanup**: When limits are exceeded:
   - Old messages are summarized
   - Summaries replace original messages
   - Multiple summaries are compressed into meta-summaries

### Error Handling

- Graceful fallback from Groq to Chutes.ai
- Connection error handling for Qdrant
- Comprehensive logging for debugging


## API Rate Limits

Be aware of rate limits for the services:
- **Groq**: Check your plan's limits
- **Chutes.ai**: Varies by subscription
- **Qdrant**: Generous limits on cloud plans

## Security Notes

- Keep your API keys secure and never commit them to version control
- Use environment variables or Streamlit secrets for production deployments
- Regularly rotate your API keys

## Troubleshooting

### Common Issues

1. **"Missing required library" error**: Install all dependencies from requirements.txt
2. **Qdrant connection failed**: Check your URL and API key
3. **AI response errors**: Verify your Groq and Chutes.ai API keys
4. **Memory issues**: Adjust memory limits in the configuration

### Logging

The application includes comprehensive logging. Check the console output for detailed information about:
- Memory operations
- API calls
- Error conditions
- Performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs for error details
- Open an issue in the repository
