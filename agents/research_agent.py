import os
from typing import List, Dict
from langchain.schema import Document
from google import genai
from google.genai import types

class ResearchAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        
        self.config = types.GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.4,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                )
            ]
        )
        print("Gemini Client for ResearchAgent initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()
    
    def generate_prompt(self, question: str, context: str) -> str:
        return f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
    
    def generate(self, question: str, documents: List[Document]) -> Dict:
        context = "\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(question, context)
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=self.config
            )
            generated_text = response.text 
            print("LLM response received.")

        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        sanitized_answer = self.sanitize_response(generated_text) if generated_text else "I cannot generate an answer."
        return {
            "answer": sanitized_answer,
            "source_documents": documents
        }