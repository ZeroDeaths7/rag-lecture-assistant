import os
from typing import List, Dict
from langchain.schema import Document
from google import genai
from google.genai import types

class VerificationAgent:
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
        print("Gemini Client for VerificationAgent initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()
    
    def generate_prompt(self, answer: str, context: str) -> str:
        return f"""
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

    def parse_verification_response(self, response_text: str) -> Dict:
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
        if not verification:
            return "Error: Unable to generate verification report."

        report = (
            f"The answer is {'supported' if verification['Supported'] == 'YES' else 'not supported'} by the context. "
            f"It is {'relevant' if verification['Relevant'] == 'YES' else 'not relevant'} to the question. "
        )

        if verification['Unsupported Claims']:
            unsupported = ', '.join(verification['Unsupported Claims'])
            report += f"Unsupported claims include: {unsupported}. "

        return report.strip()

    def check(self, answer: str, context_docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = self.generate_prompt(answer, context)

        # NEW CALL SYNTAX
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=self.config
        )
        
        sanitized_response = self.sanitize_response(response.text)
        verification = self.parse_verification_response(sanitized_response)
        report = self.format_verification_report(verification)
        
        return report