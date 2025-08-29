import json
import google.generativeai as genai
from langchain.schema import Document
from typing import List, Dict
import os

class VerificationAgent:
    def __init__(self):
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config={
                "max_output_tokens": 512,
                "temperature": 0.4,
            }
        )
        print("Gemini model for VerificationAgent initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping excess whitespace.
        """
        return response_text.strip()
    
    def generate_prompt(self, answer: str, context: str) -> str:

        """
        Generate a structured prompt for the LLM to verify the answer against the context.
        """
        prompt = f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

        **Instructions:**
        - Verify the following answer against the provided context.
        - Check for:
        1. Direct/indirect factual support (YES/NO)
        2. Unsupported claims (list any if present)
        3. Contradictions (list any if present)
        4. Relevance to the question (YES/NO)
        - Provide additional details or explanations where relevant.
        - Respond in the exact format specified below without adding any unrelated information.

        **Format:**
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]

        **Answer:** {answer}
        **Context:**
        {context}

        **Respond ONLY with the above format.**
        """
        return prompt
    

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        Parse the LLM's verification response into a structured dictionary.
        """
        try:
            lines = response_text.strip().split("\n")
            result = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": []
            }

            current_key = None
            for line in lines:
                line = line.strip()
                if line.startswith("Supported:"):
                    result["Supported"] = line.split(":", 1)[1].strip()
                elif line.startswith("Unsupported Claims:"):
                    current_key = "Unsupported Claims"
                elif line.startswith("Contradictions:"):
                    current_key = "Contradictions"
                elif line.startswith("Relevant:"):
                    result["Relevant"] = line.split(":", 1)[1].strip()
                elif line.startswith("Additional Details:"):
                    current_key = "Additional Details"
                elif current_key:
                    result[current_key].append(line)
            return result
        except Exception as e:
            print(f"Error parsing verification response: {e}")
            return None
        
    
    def format_verification_report(self, verification: Dict) -> str:
        """
        format dict into a paragraph
        """

        if not verification:
            return "Error: Unable to generate verification report."

        report = (
            f"The answer is {'supported' if verification['Supported'] == 'YES' else 'not supported'} by the context. "
            f"It is {'relevant' if verification['Relevant'] == 'YES' else 'not relevant'} to the question. "
        )

        if verification['Unsupported Claims']:
            unsupported = ', '.join(verification['Unsupported Claims'])
            report += f"Unsupported claims include: {unsupported}. "

        if verification['Contradictions']:
            contradictions = ', '.join(verification['Contradictions'])
            report += f"Contradictions found: {contradictions}. "

        if verification['Additional Details']:
            additional = ' '.join(verification['Additional Details'])
            report += f"Additional details: {additional}"

        return report.strip()
    


    def check(self, answer: str, context_docs: List[Document]) -> str:
        """
        Verify the provided answer against the context documents.
        """
        context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = self.generate_prompt(answer, context)

        response = self.model.generate_text(prompt) #calling model to generate response
        sanitized_response = self.sanitize_response(response.text)
        
        verification = self.parse_verification_response(sanitized_response)
        report = self.format_verification_report(verification)
        
        return report
