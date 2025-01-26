import io
import os
import traceback
import base64

import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pytesseract  # For OCR
from PIL import Image


from langchain_community.embeddings import OpenAIEmbeddings
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
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# 4. PDF & Image helper functions
def extract_text_from_pdf(file) -> str:
    """Extract text from a (non-scanned) PDF using PyMuPDF."""
    extracted_text = ""
    try:
        file.seek(0)
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                # If text is empty, fallback to OCR
                if not page_text.strip():
                    pix = page.get_pixmap()  # Render page as an image
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_text = pytesseract.image_to_string(img)
                extracted_text += page_text + "\n\n"
    except Exception as e:
        traceback.print_exc()
    return extracted_text.strip()


# def preprocess_image(image_file):
#     """Resize and preprocess the image to reduce size."""
#     try:
#         image = Image.open(image_file)
#         image = image.convert("RGB")  # Ensure RGB format
#         image = image.resize((128, 128))  # Resize to reduce size
#         byte_stream = io.BytesIO()
#         image.save(byte_stream, format="JPEG", quality=50)  # Compress image
#         byte_stream.seek(0)
#         return byte_stream.getvalue()
#     except Exception as e:
#         traceback.print_exc()
#         return None


# def generate_image_embedding(image_data):
#     """Generate an embedding for preprocessed image data."""
#     try:
#         if not image_data:
#             return None
#         # Assuming embeddings.embed_documents is used correctly

#         encoded_image = base64.b64encode(image_data).decode('utf-8')
        
#         # Use the encoded string as a placeholder for embeddings
#         return embeddings.embed_documents([encoded_image])[0]
    
#         # text_representation = image_data.decode("latin-1")
#         # return embeddings.embed_documents([text_representation])[0]
#     except Exception as e:
#         traceback.print_exc()
#         return "[Error in generating image embedding]"
    
# def encode_image_as_base64(image_file) -> str:
#     """Base64-encode an image for passing as placeholder text."""
#     try:
#         contents = image_file.read()
#         encoded = base64.b64encode(contents).decode("utf-8")
#         return f"[IMAGE FILE (base64, first 300 chars)]: {encoded[:300]}..."
#     except Exception as e:
#         traceback.print_exc()
#         return "[IMAGE FILE ENCODE ERROR]"


def extract_text_from_image(image_file) -> str:
    """
    Extract text from an image using Tesseract OCR.
    """
    try:
        image = Image.open(image_file)  # Open image using PIL
        extracted_text = pytesseract.image_to_string(image)  # Perform OCR
        return extracted_text.strip()
    except Exception as e:
        traceback.print_exc()
        return "[Error in OCR text extraction]"

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
    if "is_loading" not in st.session_state:
        st.session_state["is_loading"] = False  # Loading state

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
        "Upload an Image/PDF , and its embedding/content will be appended to your prompt:",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf", "webp"],
    )

    # Placeholder for loading spinner
    loading_placeholder = st.empty()

    #  # Disable send button and show loader
    # send_button_disabled = st.session_state.get("is_loading", False)

    if st.button("Send"):
        st.session_state["is_loading"] = True  # Set loading state to True
        with loading_placeholder.container():
            st.write("Processing your request, please wait...")
            st.spinner("Loading...")  # Display loading spinner

        # Combine user question + file data
        combined_input = user_input.strip()
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    combined_input += f"\n\n[PDF TEXT EXTRACTED]:\n{pdf_text}"
            else:
                ocr_text = extract_text_from_image(uploaded_file)  # Extract text using OCR
                if ocr_text:
                    combined_input += f"\n\n[TEXT EXTRACTED FROM IMAGE]:\n{ocr_text}"
                # encoded = encode_image_as_base64(uploaded_file)
                # combined_input += f"\n\n[ENCODED IMAGE AS BASE64]: {encoded}"
                # image_data = preprocess_image(uploaded_file)
                # if image_data:
                #     embedding = generate_image_embedding(image_data)
                #     if embedding:
                #         combined_input += f"\n\n[IMAGE EMBEDDING]: {embedding[:300]}..."

        # Get AI response
        try:
            response = run_chat(combined_input)
            st.success("Response generated successfully!")
            # st.write(response)
        except Exception as e:
            traceback.print_exc()
            response = f"Error: {e}"
            # store error so conversation remains consistent
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(response)
            # st.error(f"An error occurred: {e}")

        finally:
            st.session_state["is_loading"] = False 
            loading_placeholder.empty() 

    #      # Show spinner while loading
    # if st.session_state["is_loading"]:
    #     st.markdown("<div style='text-align: center;'><div class='loader'></div></div>", unsafe_allow_html=True)
    #     st.markdown(
    #         """
    #         <style>
    #         .loader {
    #             border: 16px solid #f3f3f3; /* Light grey */
    #             border-top: 16px solid #3498db; /* Blue */
    #             border-radius: 50%;
    #             width: 120px;
    #             height: 120px;
    #             animation: spin 2s linear infinite;
    #         }
    #         @keyframes spin {
    #             0% { transform: rotate(0deg); }
    #             100% { transform: rotate(360deg); }
    #         }
    #         </style>
    #         """,
    #         unsafe_allow_html=True,
    #     )

        st.rerun()


if __name__ == "__main__":
    main()