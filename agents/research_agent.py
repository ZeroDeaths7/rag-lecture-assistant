from config.settings import Settings
import google.generativeai as genai
import os
from typing import List, Dict
from langchain.schema import Document

class ResearchAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        generation_configs = {
            "max_output_tokens": 512,
            "temperature": 0.4,
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config=generation_configs
        )
        print("Gemini model for ResearchAgent initialized successfully.")

    
    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping excess whitespace.
        """
        return response_text.strip()
    
    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual. N
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt
    
    def generate(self, question: str, documents: List[Document]) -> Dict:

        context = "\n".join([doc.page_content for doc in documents])

        prompt = self.generate_prompt(question, context)
        response = self.model.generate(prompt)
        
        try:
            response = self.model.generate_content(prompt)
            
            generated_text = response.text 
            print("LLM response received.")

        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e
        


        try:
            sanitized_answer = self.sanitize_response(generated_text) if generated_text else "I cannot generate an answer."
            print(f"Sanitized Answer: {sanitized_answer}")
            return {
                "answer": sanitized_answer,
                "source_documents": documents
            }
        
        except Exception as e:
            print(f"Error during response sanitization: {e}")
            return {
                "answer": "Failed to generate answer due to a processing error.",
                "source_documents": documents
            }
        

        