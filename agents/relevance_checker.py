import os
from google import genai
from google.genai import types
from config.settings import settings

class RelevanceChecker:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        
        self.config = types.GenerateContentConfig(
            max_output_tokens=200,
            temperature=0.1,
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
        print("Gemini Client for RelevanceChecker initialized successfully.")


    def check(self, question: str, retriever, k = 3) -> str:
        top_docs = retriever.invoke(question)

        if not top_docs:
            print("No documents returned. Classifying as NO_MATCH.")
            return "NO_MATCH"
        
        document_content = " ".join([doc.page_content for doc in top_docs[:k]])

        prompt = f"""
        You are an AI relevance checker.
        
        **Instructions:**
        - Check if the provided Passages contain information relevant to the Question.
        - Respond with ONE label: CAN_ANSWER, PARTIAL, or NO_MATCH.
        - Do not provide explanations.
        
        **Question:** {question}
        
        **Passages:** {document_content}
        
        **Label:**
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=self.config
            )
            
            if not response.text:
                 print("Warning: Model returned no content.")
                 return "NO_MATCH"
                 
            generated_text = response.text 
            print("LLM response received.")

        except Exception as e:
            print(f"Error during model inference: {e}")
            return "NO_MATCH"

        try:
            import re
            clean_response = re.sub(r'[^A-Z_]', '', generated_text.strip().upper())
            print(f"LLM response: {clean_response}")

        except Exception as e:
            print(f"Unexpected response structure: {e}")
            return "NO_MATCH"
        
        if "CAN_ANSWER" in clean_response:
            return "CAN_ANSWER"
        elif "PARTIAL" in clean_response:
            return "PARTIAL"
        else:
            return "NO_MATCH"