

# RAG Chatbot with Streamlit

This repository contains a **Retrieval-Augmented Generation (RAG) chatbot** built using **Streamlit**, **Sentence Transformers**, and the **Hugging Face Transformers pipeline**. The chatbot leverages Wikipedia as its knowledge source and provides answers to user queries by retrieving relevant context and generating responses.

The interface is fully interactive, featuring a conversational UI with AI and user message bubbles, input handling, and a typing animation.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [How It Works](#how-it-works)
* [Key Components](#key-components)
* [Customization](#customization)


---

## Features

* Interactive chat interface powered by **Streamlit**.
* Retrieval of relevant information from **Wikipedia**.
* Question answering using a pre-trained **RoBERTa model** (`deepset/roberta-base-squad2`).
* Semantic search with **Sentence Transformers** (`all-MiniLM-L6-v2`).
* Typing animation to simulate AI response streaming.
* Dark-themed, responsive UI with scrollable messages and user/AI chat bubbles.
* Session management to retain conversation history.

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd rag-chatbot
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```



**Required packages include:**

* `streamlit`
* `torch`
* `transformers`
* `sentence-transformers`
* `wikipedia`
* `pyngrok` (optional, if running via ngrok)

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open the app in your browser (typically at `http://localhost:8501`).

3. Ask any question in the input box. The chatbot will:

   * Retrieve relevant Wikipedia articles.
   * Split content into manageable chunks.
   * Compute embeddings for semantic similarity.
   * Select top relevant chunks and generate an answer using the QA pipeline.

4. AI responses appear with a **typing animation**, simulating a real-time response.

---

## How It Works

The chatbot combines **retrieval-based search** with **generative question answering**:

1. **Query Input**: User enters a question.
2. **Wikipedia Retrieval**: The `fetch_wikipedia` function searches Wikipedia for relevant articles and extracts paragraphs.
3. **Chunking**: Paragraphs are split into smaller chunks (`split_into_chunks`) for embedding calculation.
4. **Semantic Search**: Using `SentenceTransformer`, the chatbot calculates embeddings for both the query and the knowledge chunks, then finds the most semantically similar chunks.
5. **QA Generation**: The top chunks are concatenated as context and passed to the Hugging Face **question-answering pipeline** to generate a response.
6. **Display**: The answer is displayed in the chat interface, word by word, with a typing animation for realism.

---

## Key Components

* **Retrieval Model**: `all-MiniLM-L6-v2` for semantic embeddings.
* **Generative Model**: `deepset/roberta-base-squad2` for question answering.
* **Session State**: Keeps track of messages and pending prompts.
* **UI Styling**: Custom dark theme using Streamlit `st.markdown` and CSS.
* **Typing Animation**: Simulates AI typing with incremental word display.
* **Auto Scroll**: JavaScript ensures the chat scrolls to the latest message.

---

## Customization

* **Change Logo**: Update the `logo_src` variable to use a custom logo.
* **Adjust Chunk Size**: Modify `split_into_chunks` parameters for larger or smaller text chunks.
* **Max Context**: Limit the context passed to the QA pipeline (`context[:2000]`) to adjust for model input length.
* **Models**: Swap the retrieval or QA models with other Sentence Transformers or Hugging Face models.

---







