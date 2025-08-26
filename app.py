import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from newspaper import Article
import re

# load env
load_dotenv()

@st.cache_resource
def init_resources():
    # streamlit secrets
    model_name = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = float(st.secrets.get("OPENAI_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", 0.3)))
    max_tokens = int(st.secrets.get("OPENAI_MAX_TOKENS", os.getenv("OPENAI_MAX_TOKENS", 1500)))
    embedding_model = st.secrets.get("OPENAI_EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    
    # api key
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    
    # set key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # llm
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # embeddings
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # vector db
    vector_path = "faiss_index"
    index_faiss = os.path.join(vector_path, "index.faiss")
    index_pkl = os.path.join(vector_path, "index.pkl")

    vectorstore = None
    if os.path.exists(index_faiss) and os.path.exists(index_pkl):
        vectorstore = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)

    # qa chain
    qa_chain = None
    if vectorstore:
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="map_reduce"
        )

    return llm, embeddings, vectorstore, qa_chain

def extract_source_quotes(llm, sources, processed_docs, query, answer):
    """Extract key quote from each source that supports the answer"""
    source_quotes = {}
    
    try:
        source_to_content = {}
        for doc in processed_docs:
            source_url = doc.metadata.get("source", "")
            if source_url:
                source_to_content[source_url] = doc.page_content
        
        for source in sources:
            if source in source_to_content:
                content = source_to_content[source]
                
                # Check if the answer indicates uncertainty
                uncertainty_indicators = ["i don't know", "i'm not sure", "unclear", "cannot determine", "no information", "not mentioned"]
                answer_lower = answer.lower()
                is_uncertain = any(indicator in answer_lower for indicator in uncertainty_indicators)
                
                if is_uncertain:
                    source_quotes[source] = "N/A"
                else:
                    quote_prompt = f"""
                    CRITICAL INSTRUCTIONS:
                    - You must extract EXACTLY ONE quote from the article below
                    - The quote must be DIRECTLY related to this specific answer: "{answer[:400]}"
                    - The quote must help answer this question: "{query}"
                    - Extract the quote word-for-word from the article text
                    - Maximum 60 characters
                    - Return ONLY the quote text with NO quotation marks, NO explanations, NO additional text
                    - If no quote directly relates to the answer, return exactly: "No relevant quote"
                    
                    Article content:
                    {content[:2000]}
                    
                    Quote:"""
                    
                    response = llm.invoke(quote_prompt)
                    quote = response.content.strip().replace('"', '').replace("'", "").replace("Quote:", "").strip()
                    
                    # Handle edge cases
                    if not quote or quote.lower() in ["no relevant quote", "no quote found", "none"]:
                        quote = "N/A"
                    elif len(quote) > 60:
                        quote = quote[:57] + "..."
                    
                    source_quotes[source] = quote

llm, embeddings, vectorstore, qa_chain = init_resources()

# UI
st.set_page_config(page_title="EquityLens", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
st.markdown("## 🔍 EquityLens : *AI-powered news analysis for smarter equity research.*")
url = "https://ghj95.github.io/portfolio//"
st.markdown(
    f"<a href='{url}' target='_blank' style='text-decoration: none; color: inherit;'>`By : Gabriel Hardy-Joseph`</a>",
    unsafe_allow_html=True,
)

# app description
def appinfo():
    with st.container(border=True):
        st.write(
            "**EquityLens** is an end-to-end LLM-powered news research tool built with *LangChain* and *OpenAI API*. "
            "It helps equity research analysts efficiently analyze financial news by retrieving, summarizing, "
            "and highlighting key market insights. Designed as a real-world NLP project, it demonstrates how large "
            "language models can streamline research workflows and transform unstructured news into actionable intelligence."
        )
    with st.container(border=True):
        st.write(
            "Try it now by adding up to three URLs and clicking on **Process**. You'll then be able to ask questions about these specific articles below."
        )

appinfo()
st.markdown("---")

# sidebar
st.sidebar.header("Articles \n (paste URLs below)")
url1 = st.sidebar.text_input("Source 1")
url2 = st.sidebar.text_input("Source 2")
url3 = st.sidebar.text_input("Source 3")

# fetch article + source
def fetch_article_document(url: str) -> Document:
    article = Article(url)
    article.download()
    article.parse()
    content = article.text
    if not content.strip():
        return None
    return Document(page_content=content, metadata={"source": url})

# Store processed documents in session state
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []

# process urls
if st.sidebar.button("Process"):
    path = "faiss_index"
    urls = [u for u in [url1, url2, url3] if u]

    if not urls:
        st.error("Please enter at least one article URL.")
    else:
        try:
            with st.spinner("Fetching article texts..."):
                docs = []
                for u in urls:
                    doc = fetch_article_document(u)
                    if doc:
                        docs.append(doc)
                    else:
                        st.warning(f"No content found for URL: {u}")

            if not docs:
                st.error("No valid article content found. Please check your URLs.")
            else:
                # Store original documents for quote extraction
                st.session_state.processed_docs = docs
                
                with st.spinner("Splitting documents into chunks..."):
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", ".", ","],
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    split_docs = text_splitter.split_documents(docs)

                with st.spinner("Embedding into vector database..."):
                    vector = FAISS.from_documents(split_docs, embeddings)
                    vector.save_local(path)

                st.success("Articles processed and indexed!")
                st.rerun()

        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")

# ask questions
vector_path = "faiss_index"
index_faiss = os.path.join(vector_path, "index.faiss")
index_pkl = os.path.join(vector_path, "index.pkl")

if os.path.exists(index_faiss) and os.path.exists(index_pkl):
    st.markdown("### 💬 Ask Questions About Your Articles")
    
    query = st.text_input("Enter your question about the processed articles:")

    if query:
        try:
            with st.spinner("Retrieving relevant chunks..."):
                vs = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
                retriever = vs.as_retriever()

            with st.spinner("Analyzing your question..."):
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, 
                    retriever=retriever
                )
                result = chain({"question": query}, return_only_outputs=True)
            
            st.markdown("### 📝 Answer")
            st.write(result["answer"])
            
            # display quotes
            if st.session_state.processed_docs:
                with st.spinner("Extracting key quotes..."):
                    key_quotes = extract_key_quotes(llm, st.session_state.processed_docs, query)
                
                if key_quotes and "Unable to extract" not in key_quotes:
                    with st.container(border=True):
                        st.markdown(key_quotes)
            
            if "sources" in result and result["sources"]:
                
                sources = result["sources"]
                
                if isinstance(sources, str):
                    sources = [s.strip() for s in sources.replace("\n", ",").split(",") if s.strip()]

                for i, src in enumerate(sources, start=1):
                    st.markdown(
                        f'<a href="{src}" target="_blank" style="display:inline-block; background-color:#F0F2F6; color:black; padding:5px 10px; margin:2px; border-radius:5px; text-decoration:none;">Source {i}</a>',
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")