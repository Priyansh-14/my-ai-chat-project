import os
import base64
import traceback

import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF

from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

# 1. Load environment variables
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# 2. System instructions (Domain = Streamlit Only)
SYSTEM_INSTRUCTIONS = """
You are a helpful AI assistant that *only* answers questions related to the Streamlit Python library.

Your scope includes:
1. Basic usage of Streamlit (e.g. installation, setup, running apps).
2. Core Streamlit functionalities (widgets, layout, theming, etc.).
3. Integrating Streamlit with other Python libraries (e.g., Pandas, NumPy, matplotlib, etc.) to display data or visualizations.
4. Advanced Streamlit topics (session states, caching, performance optimization, custom components, etc.).
5. Code examples, explanations, and best practices specifically for building Streamlit apps.

If the user asks anything outside Streamlit or out of scope, you must respond with:
"It is outside the bounds of this chat bot."

Do not provide any other information if the question is out of scope.
Stay within these instructions for the entire conversation.

Your responses:
- Should focus on Streamlit usage, best practices, tips, and troubleshooting.
- May include code snippets, sample code blocks, or references to official Streamlit documentation, as long as they pertain strictly to Streamlit.
- Should remain concise, accurate, and on topic.
- If the user question overlaps with other Python topics, only answer the portion that directly concerns Streamlit usage or integration. For the parts not related to Streamlit, respond with: 
  "It is outside the bounds of this chat bot."
"""

# 3. Initialize the ChatOpenAI LLM

llm = ChatOpenAI(
    api_key=openai_api_key,
    model_name="gpt-4o-mini",
    temperature=0.5,
)


# 4. PDF & Image helper functions
def extract_text_from_pdf(file) -> str:
    """Extract text from a (non-scanned) PDF using PyMuPDF."""
    extracted_text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                extracted_text += page.get_text()
    except Exception as e:
        traceback.print_exc()
    return extracted_text.strip()

def encode_image_as_base64(image_file) -> str:
    """Base64-encode an image for passing as placeholder text."""
    try:
        contents = image_file.read()
        encoded = base64.b64encode(contents).decode("utf-8")
        return f"[IMAGE FILE (base64, first 300 chars)]: {encoded[:300]}..."
    except Exception as e:
        traceback.print_exc()
        return "[IMAGE FILE ENCODE ERROR]"


# 5. Our core chat function
def run_chat(user_input: str) -> str:
    """
    - Build a list of messages:
        1) system message with instructions
        2) all prior user & AI messages
        3) the new user message
    - Call the LLM with `predict_messages()`
    - Return the AI's text
    """
    messages = []

    # a) Start with system instructions
    messages.append(SystemMessage(content=SYSTEM_INSTRUCTIONS))

    # b) Add the conversation history so far
    #    st.session_state["past"] holds user messages
    #    st.session_state["generated"] holds AI responses
    for user_msg, ai_msg in zip(st.session_state["past"], st.session_state["generated"]):
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=ai_msg))

    # c) Add the new user message
    messages.append(HumanMessage(content=user_input))

    # d) Get the AI response
    response_msg = llm.invoke( messages)
    assistant_text = response_msg.content

    # e) Store new messages in session_state
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(assistant_text)

    return assistant_text


# 6. Streamlit app
def main():
    st.set_page_config(page_title="My App - Home", page_icon="🤖")

    st.title("Streamlit Chatbot")
    st.write("Use the sidebar to navigate between pages.")

    # st.balloons()

    if "past" not in st.session_state:
        st.session_state["past"] = []       # user inputs
    if "generated" not in st.session_state:
        st.session_state["generated"] = []  # AI responses

    st.write("This bot **only** answers questions about **Streamlit**. "
             "If you ask out of scope, it will refuse.")

    # Display existing conversation
    for user_msg, ai_msg in zip(st.session_state["past"], st.session_state["generated"]):
        st.markdown(f"**User:** {user_msg}")
        st.markdown(f"**AI:** {ai_msg}")

    # New user input
    user_input = st.text_input("Your question about Streamlit:", "")

    # Optional file upload
    uploaded_file = st.file_uploader(
        "Upload an Image/PDF (optional), and its content/placeholder will be appended to your prompt:",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf", "webp"],
    )

    if st.button("Send"):
        # Combine user question + file data
        combined_input = user_input.strip()
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    combined_input += f"\n\n[PDF TEXT EXTRACTED]:\n{pdf_text}"
            else:
                encoded = encode_image_as_base64(uploaded_file)
                combined_input += f"\n\n{encoded}"

        # Get AI response
        try:
            response = run_chat(combined_input)
        except Exception as e:
            traceback.print_exc()
            response = f"Error: {e}"
            # store error so conversation remains consistent
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(response)

        st.rerun()


if __name__ == "__main__":
    main()