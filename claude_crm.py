"""
CRM Manager AI Agent — built with LangGraph + OpenAI
Persona: A sharp, proactive CRM manager focused on customer follow-ups.
"""
from dotenv import load_dotenv
import os
import sys
import uuid
load_dotenv()

# Windows terminal UTF-8 fix
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Annotated, TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


api_key = os.getenv("OPENAI_API_KEY")  # .env se _KEY wali value aayegi


# ── 1. State ─────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_context: dict        # optional CRM data injected per session
    follow_up_actions: list[str]  # actions the agent has suggested this session


# ── 2. CRM Manager Persona ───────────────────────────────────────────────────

CRM_SYSTEM_PROMPT = """You are GreySpaces Retail Creative Strategist—the senior-most strategist in the pipeline. You operate at the intersection of Business Strategy, Category Understanding, and Consumer Psychology. You act as the bridge between the business brief and research intelligence.

Objective: Your goal is to understand the user's business needs in a focused and insightful way, then orchestrate downstream workflows to produce a clear, structured, creative-ready strategic output.

Tone
Engagement Phase: Insightful, natural, strategic, and supportive. 

🧠 Strategic Constraints

1. Non-Creative by Design: You do NOT propose design ideas, visuals, layouts, or themes during the strategy phase.
2. Signal-Driven: Work strictly with user-provided insights and tool-returned data.
3. ask one question at a time. DO not ask many questions in a single response. 

CURRENT DATE: {current_date}

{customer_context_block}

Always end responses with a clear, specific next action if one is relevant. \
Format it as: "→ Next action: [action]"
"""


def build_system_prompt(customer_context: dict) -> str:
    """Dynamically inject CRM context into the system prompt."""
    if customer_context:
        lines = ["LOADED CUSTOMER CONTEXT:"]
        for k, v in customer_context.items():
            lines.append(f"  • {k}: {v}")
        context_block = "\n".join(lines)
    else:
        context_block = "No specific customer context loaded. Working in general mode."

    return CRM_SYSTEM_PROMPT.format(
        current_date=datetime.now().strftime("%A, %B %d, %Y"),
        customer_context_block=context_block,
    )


# ── 3. Graph Nodes ───────────────────────────────────────────────────────────

def crm_agent_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """Core agent node — calls the LLM with full conversation history."""

    system_prompt = build_system_prompt(state.get("customer_context", {}))

    messages_for_llm = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.invoke(messages_for_llm)

    # Extract any follow-up actions mentioned (simple heuristic)
    actions = state.get("follow_up_actions", [])
    if "→ Next action:" in response.content:
        action_line = [
            line for line in response.content.split("\n")
            if "→ Next action:" in line
        ]
        if action_line:
            actions = actions + [action_line[0].replace("→ Next action:", "").strip()]

    return {
        "messages": [response],
        "follow_up_actions": actions,
    }


# ── 4. Build the Graph ───────────────────────────────────────────────────────

def build_crm_graph(openai_api_key: str, model: str = "gpt-4o") -> StateGraph:
    """
    Construct and compile the LangGraph agent.

    Args:
        openai_api_key: Your OpenAI API key.
        model: OpenAI model to use (default: gpt-4o).

    Returns:
        A compiled LangGraph app with in-memory checkpointing.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0.7,          # a bit of personality
        api_key=openai_api_key,
    )

    # Wire up the node with the LLM baked in
    def agent_node(state: AgentState) -> AgentState:
        return crm_agent_node(state, llm)

    graph = StateGraph(AgentState)
    graph.add_node("crm_manager", agent_node)

    graph.add_edge(START, "crm_manager")
    graph.add_edge("crm_manager", END)

    # MemorySaver = in-memory persistence per thread_id
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app


# ── 5. Convenience Chat Runner ───────────────────────────────────────────────

class CRMManagerAgent:
    """
    High-level wrapper around the LangGraph CRM agent.

    Usage:
        agent = CRMManagerAgent(api_key="sk-...")
        agent.load_customer({"Customer": "Acme Corp", "Last contact": "22 days ago", ...})
        reply = agent.chat("What should I do about Acme?")
        print(reply)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        thread_id: str = "default-session",
    ):
        self.app = build_crm_graph(api_key, model)
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}
        self.customer_context: dict = {}

    def load_customer(self, context: dict) -> None:
        """
        Load CRM data for a specific customer into the agent's context.

        Example:
            agent.load_customer({
                "Customer": "Acme Corp",
                "ARR": "$48,000",
                "Last contact": "22 days ago",
                "Contact name": "Sarah Chen, VP Sales",
                "Last interaction": "Demo call — positive, requested pricing",
                "Open tasks": "Send proposal, intro to legal team",
            })
        """
        self.customer_context = context

    def chat(self, user_message: str) -> str:
        """Send a message and get the CRM manager's response."""
        current_actions = self.get_action_log()
        result = self.app.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "customer_context": self.customer_context,
                "follow_up_actions": current_actions,
            },
            config=self.config,
        )
        return result["messages"][-1].content

    def get_action_log(self) -> list[str]:
        """Return all follow-up actions suggested in this session."""
        state = self.app.get_state(config=self.config)
        return state.values.get("follow_up_actions", [])

    def new_session(self, thread_id: str | None = None) -> None:
        """Start a fresh conversation (new thread = clean memory)."""
        self.thread_id = thread_id or str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}





# adding cli.py code
# ── CHAT LOOP ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  CRM Manager AI -- Jordan")
    print("  Type 'quit' to exit")
    print("=" * 50 + "\n")

    # API key .env se aa jayegi
    agent = CRMManagerAgent(api_key=api_key, model="gpt-4o")

    # Optional: customer context load karo
    agent.load_customer({
        "Customer": "Acme Corp",
        "ARR": "$48,000",
        "Contact name": "Sarah Chen",
        "Last contact": "22 days ago",
        "Last interaction": "Demo call — requested pricing",
        "Churn risk": "Medium",
    })

    print("Jordan (CRM Manager): Initializing...\n")
    greeting = agent.chat("Hey Jordan, introduce yourself briefly.")
    print(f"Jordan: {greeting}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye! Don't forget to follow up!")
            break
        reply = agent.chat(user_input)
        print(f"\nJordan: {reply}\n")        