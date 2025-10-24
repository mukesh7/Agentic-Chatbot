import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Any, Dict, List, Tuple

class DisplayResultStreamlit:
    def __init__(self, usecase: str, graph: Any, user_message: str):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    # -- helpers -----------------------------------------------------------
    def _to_simple_dict(self, msg: Any) -> Dict[str, str]:
        """
        Normalize a message (AIMessage/HumanMessage tuple or dict) to a simple dict:
          {"role": "user"|"assistant", "content": "..."}
        """
        # LangChain message objects
        if hasattr(msg, "content") and hasattr(msg, "__class__"):
            clsname = msg.__class__.__name__.lower()
            if "human" in clsname:
                role = "user"
            elif "ai" in clsname or "assistant" in clsname:
                role = "assistant"
            elif "tool" in clsname:
                role = "tool"
            else:
                role = "assistant"
            return {"role": role, "content": getattr(msg, "content", "")}

        # tuple like ("user","hi")
        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
            role, content = msg[0], msg[1]
            return {"role": role, "content": content}

        # dict like {"role": "...", "content": "..."}
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type") or "assistant"
            content = msg.get("content") or msg.get("message") or ""
            return {"role": role, "content": content}

        # fallback: stringify
        return {"role": "assistant", "content": str(msg)}

    def _history_to_tuples(self, history: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """Convert session_state history (list of dicts) to list of (role, content) tuples."""
        return [(m["role"], m["content"]) for m in history]

    # -- main --------------------------------------------------------------
    def display_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message

        # init history as simple dicts in session_state
        if "messages" not in st.session_state:
            st.session_state.messages = []  # list of {"role":..., "content":...}

        # append user message to history
        st.session_state.messages.append({"role": "user", "content": user_message})

        # render existing history first
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        # only stream for Basic Chatbot
        if usecase == "Basic Chatbot":
            # Build state expected by graph: use tuple list format
            graph_input = {"messages": self._history_to_tuples(st.session_state.messages)}

            assistant_partial = ""  # accumulate streaming content

            # Stream events from the graph
            try:
                for event in graph.stream(graph_input):
                    # event is typically a dict mapping node_name -> node_state (dict)
                    for node_state in event.values():
                        # node_state may contain "messages" in variant forms
                        msgs = node_state.get("messages") if isinstance(node_state, dict) else None
                        if not msgs:
                            # try the whole node_state if it's actually a message
                            msgs = node_state

                        # If msgs is a single message object, wrap it
                        if msgs and not isinstance(msgs, (list, tuple)):
                            msgs = [msgs]

                        # iterate over messages and display latest assistant text if present
                        for m in msgs or []:
                            normalized = self._to_simple_dict(m)
                            # if it's assistant content, show/update live
                            if normalized["role"] in ("assistant", "ai", "bot"):
                                assistant_partial = normalized["content"]

                                # For streaming UX: update a single assistant chat message
                                # Streamlit doesn't support editing a chat message in place easily,
                                # so we simply write the current partial output (may duplicate).
                                with st.chat_message("assistant"):
                                    st.write(assistant_partial)

                # After streaming completes, store the final assistant message
                if assistant_partial:
                    st.session_state.messages.append({"role": "assistant", "content": assistant_partial})

            except Exception as e:
                # graceful fallback: show error and keep history intact
                st.error(f"Streaming error: {e}")
