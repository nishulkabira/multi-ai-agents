
import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from web_operations import serp_search, reddit_search_api, reddit_post_retrieval
from prompts import (PromptTemplates, 
                     get_google_analysis_messages, 
                     get_bing_analysis_messages, 
                     get_reddit_analysis_messages,
                     get_reddit_url_analysis_messages, 
                     get_synthesis_messages
                     )

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str | None
    google_results: str | None
    bing_results: str | None
    reddit_results: str| None
    selected_reddit_url: list[str] | None
    reddit_post_data: list | None
    google_analysis: str | None
    bing_analysis: str | None
    reddit_analysis: str | None
    final_answer: str | None


class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] =Field(description = "List of Reddit URLs that contain valuable information for answering the user's question")


def extract_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    return str(content)



def google_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Google for: {user_question}")

    google_results = serp_search(user_question, engine="google")
     
    return {"google_results": google_results}

def bing_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Bing for: {user_question}")

    bing_results = serp_search(user_question, engine="bing")
     

    return {"bing_results": bing_results}

def reddit_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Reddit for: {user_question}")

    reddit_results = reddit_search_api(user_question, num_of_posts=30)
    print(reddit_results)

    return {"reddit_results": reddit_results}

def analyze_reddit_posts(state: State):
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")

    if not reddit_results:
        print("No Reddit results found.")
        return {"selected_reddit_url": []}
    
    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, reddit_results)

    try:
        analysis = structured_llm.invoke(messages)
        selected_urls = analysis.selected_urls

        print("Selected Reddit URLs:")
        for i, url in enumerate(selected_urls, start=1):
            print(f"Selected Reddit URL {i}: {url}")
    
    except Exception as e:
        print(f"Error during Reddit URL analysis: {e}")
        selected_urls = []

    
    return {"selected_reddit_url": selected_urls}

def retrieve_reddit_posts(state: State):
    selected_reddit_url = state.get("selected_reddit_url", [])

    if not selected_reddit_url:
        print("No Reddit URLs selected for detailed retrieval.")
        return {"reddit_post_data": []}

    print(f"Retrieving Reddit post data for {len(selected_reddit_url)} URLs")
    reddit_post_data = reddit_post_retrieval(selected_reddit_url)

    if not reddit_post_data:
        print("No Reddit post data retrieved.")
        return {"reddit_post_data": []}

    print(f"Retrieved {reddit_post_data.get('total_retrieved', 0)} Reddit comments")
    return {"reddit_post_data": reddit_post_data.get("comments", [])}

def analyze_google_results(state: State):
    user_question = state.get("user_question", "")
    google_results = state.get("google_results", "")

    print("Analyzing Google search results...")
    messages = get_google_analysis_messages(user_question, google_results)
    reply = llm.invoke(messages)

    return {"google_analysis": extract_text(reply)}

def analyze_bing_results(state: State):
    user_question = state.get("user_question", "")
    bing_results = state.get("bing_results", "")

    print("Analyzing Bing search results...")
    messages = get_bing_analysis_messages(user_question, bing_results)
    reply = llm.invoke(messages)

    return {"bing_analysis": extract_text(reply)}

def analyze_reddit_results(state: State):
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", [])

    print("Analyzing Reddit discussions...")
    messages = get_reddit_analysis_messages(user_question, reddit_results, reddit_post_data)
    reply = llm.invoke(messages)

    return {"reddit_analysis": extract_text(reply)}

def synthesize_analyses(state: State):
    user_question = state.get("user_question", "")
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    print("Synthesizing final answer...")
    messages = get_synthesis_messages(user_question, google_analysis, bing_analysis, reddit_analysis)
    reply = llm.invoke(messages)
    final_answer = extract_text(reply)

    return {
        "final_answer": final_answer,
        "messages": [{"role": "assistant", "content": final_answer}],
    }

graph_builder = StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("analyze_reddit_posts", analyze_reddit_posts)
graph_builder.add_node("retrieve_reddit_posts", retrieve_reddit_posts)
graph_builder.add_node("analyze_google_results", analyze_google_results)
graph_builder.add_node("analyze_bing_results", analyze_bing_results)
graph_builder.add_node("analyze_reddit_results", analyze_reddit_results)
graph_builder.add_node("synthesize_analyses", synthesize_analyses)

graph_builder.add_edge(START, "google_search")
graph_builder.add_edge(START, "bing_search")
graph_builder.add_edge(START, "reddit_search")

graph_builder.add_edge("google_search", "analyze_reddit_posts")
graph_builder.add_edge("bing_search", "analyze_reddit_posts")
graph_builder.add_edge("reddit_search", "analyze_reddit_posts")

graph_builder.add_edge("analyze_reddit_posts", "retrieve_reddit_posts")

graph_builder.add_edge("retrieve_reddit_posts", "analyze_google_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_bing_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_reddit_results")

graph_builder.add_edge("analyze_google_results", "synthesize_analyses")
graph_builder.add_edge("analyze_bing_results", "synthesize_analyses")
graph_builder.add_edge("analyze_reddit_results", "synthesize_analyses")

graph_builder.add_edge("synthesize_analyses", END)

graph = graph_builder.compile()

def run_chatbot():
    print("Multi-Source Research Agent")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Ask me anything: ")
        if user_input.lower() == "exit":
            print("Bye :)")
            break

        state = {
            "messages": [{"role": "user", "content": user_input}],
            "user_question": user_input,
            "google_results": None,
            "bing_results": None,
            "reddit_results": None,
            "selected_reddit_url": None,
            "reddit_post_data": None,
            "google_analysis": None,
            "bing_analysis": None,
            "reddit_analysis": None,
            "final_answer": None,
        }

        print ("\nStarting parallel research process...")
        print("Launching Google, Bing, and Reddit searches...\n")
        final_state = graph.invoke(state)

        if final_state.get("final_answer"):
            print(f"\nFinal Answer:\n{final_state['final_answer']}")

        print("-" * 80)

if __name__ == "__main__":
    run_chatbot()