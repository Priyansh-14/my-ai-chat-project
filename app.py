import os
import traceback
from dotenv import load_dotenv # type: ignore
import streamlit as st # type: ignore
import openai # type: ignore
import pytesseract # type: ignore
from PIL import Image # type: ignore
import io

# Load environment variables (e.g., OPENAI_API_KEY) from .env
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# openai.api_key = openai_api_key

client = openai.OpenAI(api_key=openai_api_key)

# Initialize session state for conversation
if "messages" not in st.session_state:
    # We'll store tuples like (role, content), e.g. ("user", "...") or ("assistant", "...")
    st.session_state["messages"] = []

def get_ai_reply(prompt: str) -> str:
    """
    Calls the OpenAI API to get a reply using the latest method in openai>=1.0.0:
    openai.Chat.completions.create(...)
    """

    # Build the conversation array from st.session_state["messages"]
    conversation = [
        {"role": msg[0], "content": msg[1]} 
        for msg in st.session_state["messages"]
    ]
    conversation.append({"role": "user", "content": prompt})

    try:
        # Using the new method name for Chat completions in openai>=1.0.0
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            # temperature=0.7,
        )
        # Extract the assistant's reply
        reply = response["choices"][0]["message"]["content"]
        return reply
    except Exception as e:
        # 4) Log the full stack trace to the terminal for debugging
        traceback.print_exc()
        return f"Error: {str(e)}"


def main():
    st.title("Basic Chat with Text & Image (OCR)")
    st.write("A minimal example of a chat system that accepts text and images.")

    # Display chat history
    for (role, content) in st.session_state["messages"]:
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**AI:** {content}")

    # Text input
    user_input = st.text_input("Enter your message:", "")

    # Image upload (optional)
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

    if st.button("Send"):
        combined_prompt = user_input.strip()

        # If an image is uploaded, perform OCR
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            extracted_text = pytesseract.image_to_string(image)
            extracted_text = extracted_text.strip()
            if extracted_text:
                combined_prompt += f"\n\n[OCR TEXT FROM IMAGE]:\n{extracted_text}"

        if not combined_prompt:
            st.warning("Please type something or upload an image to extract text from.")
        else:
            # Add user message
            st.session_state["messages"].append(("user", combined_prompt))

            # Get AI reply
            ai_reply = get_ai_reply(combined_prompt)

            # Add AI reply
            st.session_state["messages"].append(("assistant", ai_reply))

            # If you wish to force a re-run to update the UI automatically,
            # use st.experimental_rerun() in older versions of Streamlit.
            # For newer versions where experimental_rerun() is removed or renamed,
            # you can simply rely on Streamlit's normal rerun after a widget event.


if __name__ == "__main__":
    main()