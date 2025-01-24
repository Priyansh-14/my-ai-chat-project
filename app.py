import os
import traceback
from dotenv import load_dotenv # type: ignore
import fitz
import streamlit as st # type: ignore
import openai # type: ignore
import pytesseract # type: ignore
from PIL import Image # type: ignore


# Load environment variables (e.g., OPENAI_API_KEY) from .env
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
# print(openai_api_key)
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# openai.api_key = openai_api_key

client = openai.OpenAI(api_key=openai_api_key)

# Initialize session state for conversation
if "messages" not in st.session_state:
    # We'll store tuples like (role, content), e.g. ("user", "...") or ("assistant", "...")
    st.session_state["messages"] = []

def get_ai_reply(prompt: str) -> str:
    # Build the conversation array from st.session_state["messages"]
    conversation = [
        {"role": msg[0], "content": msg[1]} 
        for msg in st.session_state["messages"]
    ]
    conversation.append({"role": "user", "content": prompt})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages= conversation,
            # temperature=0.7,
        )

        # Extract the assistant's reply
        reply = completion.choices[0].message.content
        # print(reply)
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

    # Image/pdf upload (optional)
    uploaded_file = st.file_uploader("Upload an image/pdf file (optional)", type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf", "webp",])

    if st.button("Send"):
        combined_prompt = user_input.strip()

        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                # Extract text from PDF
                pdf_text = ""
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    for page in doc:
                        pdf_text += page.get_text()

                pdf_text = pdf_text.strip()
                if pdf_text:
                    combined_prompt += f"\n\n[EXTRACTED TEXT FROM PDF]:\n{pdf_text}"
                else:
                    st.warning("No text found in PDF. Try the OCR approach if it's scanned.")
            else:
                # Handle as an image
                image = Image.open(uploaded_file)
                extracted_text = pytesseract.image_to_string(image).strip()
                if extracted_text:
                    combined_prompt += f"\n\n[OCR TEXT FROM IMAGE]:\n{extracted_text}"

        if not combined_prompt:
            st.warning("Please type something or upload a file to extract text from.")
        else:
                # Add user message
                st.session_state["messages"].append(("user", combined_prompt))

                # Get AI reply
                ai_reply = get_ai_reply(combined_prompt)

                # Add AI reply
                st.session_state["messages"].append(("assistant", ai_reply))

                st.rerun()

if __name__ == "__main__":
    main()