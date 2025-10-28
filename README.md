## Chat Bot (Streamlit)

A minimal chat application built with Streamlit. It provides a built-in rule-based bot and optional OpenAI integration if you supply an API key.

### Prerequisites
- Python 3.9+
- Windows PowerShell (commands below are for Windows)

### Setup
1. Create and activate a virtual environment:
   
   ```powershell
   cd C:\code\library-domain-services
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r chatbot\requirements.txt
   ```

3. (Optional) Enable OpenAI provider by setting your API key. You can either set an environment variable or create a `.env` file inside `chatbot/`.

   - Environment variable (current shell only):
     ```powershell
     $env:OPENAI_API_KEY = "sk-..."
     ```

   - Or create `chatbot/.env` with this content:
     ```
     OPENAI_API_KEY=sk-...
     ```

### Run the app

```powershell
streamlit run chatbot/app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`). Use the sidebar to switch between the Rule-based bot and the OpenAI provider.


