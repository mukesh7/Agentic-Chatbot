import streamlit as st
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI

from src.langgraphagenticai.llms.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Failed to load user input from the UI")
        return
    
    user_message = st.chat_input("Enter your message:")

    if user_message:
        try:
            obj_llm_config = GroqLLM(user_input)
            model = obj_llm_config.get_llm_model()

            if not model:
                st.error("llm model couldn't be initialized")
                return
            usecase = user_input.get("selected_usecase")

            if not usecase:
                st.error("no usecase selected")
                return
            
            graph_builder = GraphBuilder(model)
            try:
                graph = graph_builder.setup_graph(usecase)
                DisplayResultStreamlit(usecase,graph,user_message).display_result_on_ui()
            except Exception as e:
                st.error(f"Error graph setup failed {e}")
                return


        except Exception as e:
            st.error(f"Error: graph set up failed {e}")
            return




