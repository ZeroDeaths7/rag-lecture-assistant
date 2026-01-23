from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker
from langchain_core.documents import Document

# NOTE: The 'EnsembleRetriever' import has been removed to fix your error.
# We now use 'Any' for the retriever type hint.

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: Any # Using Any avoids the import error


class AgentWorkflow: 
    
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        
    
    def create_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("research", self.research_step)
        workflow.add_node("verify", self.verifier_step)
        workflow.add_node("check_relevance", self.relevance_checker_step)

        workflow.set_entry_point("check_relevance")

        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges("verify", self.after_verification, {"re_research": "research", "end": END})
        workflow.add_conditional_edges("check_relevance", self.after_relevance, {"re_research": "research", "irrelevant": END}) 

        return workflow.compile()
    

    def research_step(self, state: AgentState) -> AgentState:
        print(f"Research step initiated with question: {state['question']}")
        result = self.researcher.generate(state['question'], state['documents'])
        return {"draft_answer": result['answer']}


    def verifier_step(self, state: AgentState) -> AgentState:
        print(f"Verification step initiated with draft answer: {state['draft_answer']}")
        result = self.verifier.check(state['draft_answer'], state['documents'])
        return {"verification_report": result}


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
                "draft_answer": "This question isn't related to the lecture content (or no data was found)."
            }


    def after_relevance(self, state: AgentState):
        decision = "re_research" if state['is_relevant'] else "irrelevant"
        print(f"After relevance check: {decision}")
        return decision

    def after_verification(self, state: AgentState):
        report = state.get('verification_report', "")
        print(f"After verification: {report}")
        
        if "Supported: NO" in report or "Relevant: NO" in report:
            return "re_research"
        else:
            return "end"