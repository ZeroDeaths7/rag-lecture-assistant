#deciding if query is not valid, valid 

import os
import google.generativeai as genai

from config import settings

class RelevanceChecker:

    def __init__(self, docs):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        generation_config = {
            "max_output_tokens": 50,    
            "temperature": 0.1,          
        }

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest", # Flash is fast and efficient for this kind of task
            generation_config=generation_config
        )
        print("Gemini model for RelevanceChecker initialized successfully.")



    def check(self, question: str, retriever, k = 3) -> str:
        """
            1. Retrieve the top-k document chunks from the global retriever.
            2. Combine them into a single text string.
            3. Pass that text + question to the LLM for classification.
            Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """

        top_docs = retriever.invoke(question)

        if not top_docs:
            print("No documents returned. Classifying as NO_MATCH.")
            return "NO_MATCH"
        
        document_content = " ".join([doc.page_content for doc in top_docs[:k]])

        prompt = f"""
        You are an AI relevance checker between a user's question and provided explanation for the lecture video snippet.

        **Instructions:**
        - Classify how well the explanation for the lecture video snippet addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.
        
        **Labels:**
        1) "CAN_ANSWER": The explanation contains enough explicit information to fully answer the question.
        2) "PARTIAL": The explanation mentions or discusses the question's topic but does not provide all the details needed for a complete answer.
        3) "NO_MATCH": The explanation does not discuss or mention the question's topic at all.


        **Important:** If the explanation mentions or references the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".
        
        **Question:** {question}
        
        **Passages:** {document_content}
        
        
        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """


        #call the llm

        try:
            response = self.model.generate_content(prompt)
            
            # The actual text is in the .text attribute
            generated_text = response.text 
            print("LLM response received.")
            # Now you can use 'generated_text'

        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        try:
            llm_response = generated_text.strip().upper()
            print(f"LLM response: {llm_response}")

        except (IndexError, KeyError) as e:
            print(f"Unexpected response structure: {e}")
            return "NO_MATCH"
        
        # Validate the response
        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        if llm_response not in valid_labels:
            print("LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
            classification = "NO_MATCH"
        else:
            print(f"Classification recognized as '{llm_response}'.")
            classification = llm_response
        return classification
        
