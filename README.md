# EquityLens

**AI-powered news analysis for smarter equity research**

EquityLens is an end-to-end LLM-powered news research tool that helps equity research analysts efficiently analyze financial news by retrieving, summarizing, and highlighting key market insights. Built with LangChain and OpenAI API, it demonstrates how large language models can streamline research workflows and transform unstructured news into actionable intelligence.

## Features

- **Multi-Article Analysis**: Process up to 3 financial news articles simultaneously
- **Intelligent Q&A**: Ask questions about processed articles using natural language
- **Key Quote Extraction**: Automatically identifies and displays impactful quotes from articles
- **Trend Indicators**: Visual sentiment indicators (Positive, Negative, Neutral)
- **Source Attribution**: Direct links to original articles with contextual quotes
- **Vector Search**: FAISS-powered semantic search for accurate information retrieval

## Live Demo

[Try EquityLens on Streamlit Cloud](https://equity-lens.streamlit.app)

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT (configurable model)
- **Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: OpenAI text-embedding-3-small
- **Article Processing**: newspaper3k

## Usage

1. **Add Article URLs**: Enter up to 3 financial news article URLs in the sidebar
2. **Process Articles**: Click "Process" to analyze and index the articles
3. **Ask Questions**: Use natural language to query the processed content
4. **Review Results**: Get answers with trend indicators, key quotes, and source links

### Example Queries
- "What are the main factors affecting stock performance?"
- "What do analysts predict for Q4 earnings?"
- "What risks are mentioned in these articles?"

## Technical Implementation

### Architecture
- **RAG Framework**: Retrieval-Augmented Generation for accurate, source-backed answers
- **Document Processing**: Recursive text splitting with optimal chunk sizes (500 chars, 50 overlap)
- **Vector Store**: FAISS indexing with OpenAI embeddings for semantic search
- **LLM Integration**: Configurable OpenAI models with RetrievalQAWithSourcesChain

### Key Components
- **Multi-document Analysis**: Processes and synthesizes insights from multiple articles
- **Source Attribution**: Maintains traceability from answers back to original sources
- **Real-time Processing**: Dynamic quote extraction and sentiment analysis
- **Error Handling**: Robust handling for failed URL fetches and API calls

## Performance Considerations

- **Caching**: Streamlit resource caching for LLM initialization
- **Token Management**: Configurable token limits for cost optimization  
- **Chunk Strategy**: Optimized text splitting for accurate retrieval
- **Model Selection**: Flexible model configuration for different use cases

## Future Enhancements

- Real-time news feed integration
- Historical sentiment tracking
- Multi-language article support
- Advanced financial metrics extraction
- Integration with financial data APIs