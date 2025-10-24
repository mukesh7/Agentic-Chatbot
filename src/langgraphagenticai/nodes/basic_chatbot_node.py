from src.langgraphagenticai.state.statedefined import State

class BasicChatbotNode:
    """basic chatbot implementation"""

    def __init__(self, model):
        self.llm = model

    def process(self, state:State)->dict:
        return {"messages":self.llm.invoke(state["messages"])}
