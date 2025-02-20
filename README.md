# Document Q&A Assistant

A Streamlit application that uses RAG (Retrieval Augmented Generation) to answer questions about PDF documents.

## Features

- PDF document upload and analysis
- Question answering with detailed reasoning
- Source reference tracking
- Clean, professional UI

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy your app from the forked repository
4. Add your GROQ_API_KEY to the Streamlit secrets
   - Go to your app settings
   - Under "Secrets", add:
     ```toml
     GROQ_API_KEY = "your_api_key_here"
     ```

## Usage

1. Upload a PDF document
2. Enter your question
3. View the AI's reasoning process and answer
4. Check sources for verification

## Requirements

See `requirements.txt` for full list of dependencies. 