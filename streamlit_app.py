import streamlit as st

from main import graph

st.set_page_config(page_title="Multi-source research agent")

st.title("Multi-source research agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

STEP_LABELS = {
    "google_search": "Searching Google...",
    "bing_search": "Searching Bing...",
    "reddit_search": "Searching Reddit...",
    "analyze_reddit_posts": "Selecting relevant Reddit posts...",
    "retrieve_reddit_posts": "Retrieving Reddit post comments...",
    "analyze_google_results": "Analyzing Google results...",
    "analyze_bing_results": "Analyzing Bing results...",
    "analyze_reddit_results": "Analyzing Reddit discussions...",
    "synthesize_analyses": "Synthesizing final answer...",
}

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    state = {
        "messages": [{"role": "user", "content": prompt}],
        "user_question": prompt,
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

    with st.chat_message("assistant"):
        final_answer = None

        with st.status("Researching...", expanded=True) as status:
            for step in graph.stream(state):
                for node_name, node_output in step.items():
                    status.write(STEP_LABELS.get(node_name, node_name))
                    if node_name == "synthesize_analyses":
                        final_answer = node_output.get("final_answer")
            status.update(label="Research complete", state="complete")

        final_answer = final_answer or "Sorry, I couldn't put together an answer."
        st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
