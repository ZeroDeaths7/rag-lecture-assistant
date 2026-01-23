import os
from google import genai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
    print(f"🔑 Using API Key: {api_key[:5]}... (verified)")
    
    try:
        client = genai.Client(api_key=api_key)
        
        print("\n Checking available models (google-genai SDK)...\n")

        pager = client.models.list()
        
        count = 0
        for model in pager:
            # We print the 'name' and 'display_name'
            print(f"ID: {model.name} | Display: {model.display_name}")
            count += 1
                
        if count == 0:
            print("No models found. Your API key might be restricted.")
            
    except Exception as e:
        print(f"\n Error listing models: {e}")