from dotenv import load_dotenv
import os
import openai

# Load .env into environment as early as possible
load_dotenv()

# Expose the key for other modules
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
