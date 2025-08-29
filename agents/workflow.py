from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    VerificationAgent: str
    is_relevant: bool
    retriever: EnsembleRetriever


class AgentWorkflow: 
    
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        
    
    def create_workflow(self):
        workflow = StateGraph[AgentState]()

        workflow.add_node("research", self.research_step)
        workflow.add_node("verify", self.verifier_step)
        workflow.add_node("check_relevance", self.relevance_checker_step)

        workflow.set_entry_point("check_relevance")

        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges("verify", self.after_verification, {"is_relevant": "research", "irrelevant": END}) #come back to this
        workflow.add_conditional_edges("check_relevance", self.after_relevance, {"re_research": "research", "end": END}) #come back

        return workflow
    

    def research_step(self, state: AgentState) -> AgentState:

        print(f"Research step initiated with question: {state['question']}")
        result = self.researcher.generate(state['question'], state['documents'])
        return state


    def verifier_step(self, state: AgentState) -> AgentState:

        print(f"Verification step initiated with draft answer: {state['draft_answer']}")
        result = self.verifier.verify(state['draft_answer'], state['documents'])
        return state


    def relevance_checker_step(self, state: AgentState) -> AgentState:

        retriever = state['retriever']
        classification = self.relevance_checker.check(question = state['question'], retriever=retriever, k = 5)
        
        if classification == "CAN_ANSWER":
            return {"is_relevant": True}

        elif classification == "PARTIAL":
            return {"is_relevant": True}
        
        else:
            return {
                "is_relevant": False,
                "draft_answer": "This question isn't related (or no data) found"
        }


    def after_relevance(self, state: AgentState):
        decision = "re_research" if state['is_relevant'] else "irrelevant"
        print(f"After relevance check: {decision}")
        return decision

    def after_verification(self, state: AgentState):
        Verification_report = state['verification_report']
        print(f"After verification: {Verification_report}")
        
        if "Supported: NO" in Verification_report or "Relevant: NO" in Verification_report:
            return "re_research"
        
        else:
            return "end"





    def full_pipeline(self, question: str, retriever: EnsembleRetriever) -> str:

        documents = retriever.invoke(question)

        initial_state: AgentState = {
            "question": question,
            "documents": documents,
            "draft_answer": "",
            "verification_report": "",
            "is_relevant": False,
            "retriever": retriever
        }

        workflow = self.create_workflow()
        final_state = workflow.run(initial_state)
        return final_state['draft_answer']