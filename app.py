import streamlit as st
import datetime


st.set_page_config(
    page_title="PawsitiveHealth AI",
    page_icon="🐾",
    layout="wide"
)


st.title("🐾 PawsitiveHealth AI Assistant")
st.caption("Your AI Pet Emergency & Health Guidance Companion")

st.sidebar.header("🐕 Pet Profile & Location")

dog_name = st.sidebar.text_input("Dog's Name", value="Buddy")
dog_breed = st.sidebar.text_input("Breed", value="Golden Retriever")
dog_dob = st.sidebar.date_input("Date of Birth", value=datetime.date(2021, 1, 1))
user_location = st.sidebar.text_input("Your Location (City/Area)", value="New Delhi")

st.session_state["pet_profile"] = {
    "name": dog_name,
    "breed": dog_breed,
    "dob": str(dog_dob),
    "location": user_location
}

st.sidebar.markdown("---")
st.sidebar.subheader("Active Profile")
st.sidebar.write(f"**Name:** {dog_name}")
st.sidebar.write(f"**Breed:** {dog_breed}")
st.sidebar.write(f"**Location:** {user_location}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": f"Hello! I am your PawsitiveHealth assistant. How can I help {dog_name} today?"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# pyrefly: ignore[missing-import]
from agent import ask_pawsitive_agent
if prompt := st.chat_input("Type your question or describe symptoms..."):
    
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
                st.markdown(prompt)
    dog_info = st.session_state.get("pet_profile", {})
    full_prompt_for_agent = (
        f"[Pet Info: Name={dog_info.get('name')}, Breed={dog_info.get('breed')}, DOB={dog_info.get('dob')}, Location={dog_info.get('location')}]\n"
        f"User Message: {prompt}"
    )

    
    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing symptoms for {dog_info.get('name')} & finding emergency options..."):
            try:
                response = ask_pawsitive_agent(full_prompt_for_agent)
                st.markdown(response)
                
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_text = f"⚠️ An error occurred while running the agent: {e}"
                st.error(error_text)