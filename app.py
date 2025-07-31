import streamlit as st
from qa_engine import ask_robotics_question

st.set_page_config(page_title="Robotics Support Assistant")
st.title("🤖 Ask Your Robotics Assistant")
st.markdown("Ask technical questions about ROS, SLAM, Sensors, etc. based on uploaded manuals and tutorials.")

query = st.text_input("Ask a question:")
if query:
    with st.spinner("Thinking..."):
        answer = ask_robotics_question(query)
        st.markdown(answer)
