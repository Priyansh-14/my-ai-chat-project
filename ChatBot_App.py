import io
import os
import traceback
from typing import List

import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pytesseract  # For OCR
from PIL import Image

# --- Updated LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage

# --- Custom In-Memory History for Session Management ---
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """A simple in-memory chat history."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

    class Config:
        arbitrary_types_allowed = True

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # In production, retrieve the history based on session_id.
    # For now, we simply return a new InMemoryHistory instance.
    return InMemoryHistory()

# --- Load Environment Variables ---
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# --- System Instructions ---
SYSTEM_INSTRUCTIONS = """
You are a highly knowledgeable and versatile AI assistant that only answers questions and provides guidance related to the Streamlit Python library. Your expertise covers every aspect of Streamlit development, including but not limited to:

1. **Basic Setup and Usage:**
   - Installation (using pip, conda, etc.) and initial configuration.
   - Creating, running, and deploying a basic Streamlit app.
   - Understanding the file structure and execution model of Streamlit applications.

2. **Core Functionalities and Widgets:**
   - Detailed explanations and examples for commonly used widgets (text inputs, buttons, sliders, checkboxes, radio buttons, select boxes, etc.).
   - Layout management including columns, containers, expanders, and sidebar configuration.
   - Theming and styling options, custom CSS, and layout customization.

3. **Advanced Features and State Management:**
   - Use of session state for managing interactive and persistent user data.
   - Caching techniques (e.g., @st.cache, @st.experimental_memo) to optimize performance.
   - Best practices for performance optimization and efficient resource usage.
   - Custom component development and integration with JavaScript frameworks.

4. **Data Display and Visualization:**
   - Displaying various data formats such as tables, dataframes, images, and charts.
   - Integration with popular visualization libraries (Matplotlib, Plotly, Altair, Bokeh, etc.).
   - Techniques for creating interactive visualizations and real-time dashboards.

5. **Integration with Other Libraries and Tools:**
   - Working with data analysis libraries like Pandas, NumPy, and SciPy.
   - Connecting to external APIs, databases, and cloud storage solutions.
   - Incorporating machine learning models and real-time data processing into Streamlit apps.

6. **Deployment and Production Considerations:**
   - Guidance on deploying Streamlit apps to various platforms (Streamlit Sharing, Heroku, AWS, Google Cloud, etc.).
   - Strategies for scaling applications, monitoring performance, and ensuring security in production.
   - Integration with CI/CD pipelines and best practices for maintainability.

7. **Troubleshooting, Debugging, and Best Practices:**
   - Interpreting and resolving common error messages and issues.
   - Tips for debugging Streamlit applications and logging practices.
   - Best practices for code organization, modular design, and maintainability.
   - Strategies for enhancing user experience and accessibility.

8. **Community, Documentation, and Future Developments:**
   - Recommendations for leveraging the Streamlit community, forums, and official documentation.
   - Insights into upcoming features, updates, and trends within the Streamlit ecosystem.
   - Examples of real-world use cases and advanced implementation patterns.

When a user requests additional examples, more code samples, or further explanation on any Streamlit-related topic, you must provide detailed, precise, and comprehensive answers that cover both fundamental concepts and edge cases. Only if the question is clearly not related to Streamlit (or its application in code and conceptual use) should you respond with:
"It is outside the bounds of this chat bot."
"""

# --- Initialize ChatOpenAI ---
llm = ChatOpenAI(
    api_key=openai_api_key,
    model_name="gpt-4o-mini",
    temperature=0.5,
)

# --- Define Prompt Template ---
# This prompt concatenates the system instructions, the chat history (keyed as "history"),
# and the current user input (as "{input}").
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_INSTRUCTIONS),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{input}")
])

# --- Create a Runnable with Message History ---
# This wraps the chain (prompt then LLM) so that it automatically maintains context.
conversation = RunnableWithMessageHistory(
    runnable=chat_prompt | llm,
    get_session_history=get_session_history
)

# --- PDF & Image Helper Functions ---
def extract_text_from_pdf(file) -> str:
    """Extract text from a (non-scanned) PDF using PyMuPDF (fallbacks to OCR if needed)."""
    extracted_text = ""
    try:
        file.seek(0)
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                # If the page has no text, use OCR
                if not page_text.strip():
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_text = pytesseract.image_to_string(img)
                extracted_text += page_text + "\n\n"
    except Exception:
        traceback.print_exc()
    return extracted_text.strip()

def extract_text_from_image(image_file) -> str:
    """Extract text from an image using Tesseract OCR."""
    try:
        image = Image.open(image_file)
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text.strip()
    except Exception:
        traceback.print_exc()
        return "[Error in OCR text extraction]"

# --- Core Chat Function ---
def run_chat(user_input: str, session_id: str) -> str:
    """
    Invokes the runnable (which automatically includes chat history)
    using the provided session_id.
    """
    response = conversation.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    # Extract the answer text from the response (assumes response has a "content" attribute)
    answer = response.content if hasattr(response, "content") else str(response)

    # Store the conversation in Streamlit session state for display.
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(answer)
    return answer

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="My App - Home", page_icon="🤖")
    st.title("Streamlit Chatbot")
    st.write("This bot **only** answers questions about **Streamlit**. If you ask out of scope, it will refuse.")

    # Initialize session state for conversation display
    if "past" not in st.session_state:
        st.session_state["past"] = []  # user inputs
    if "generated" not in st.session_state:
        st.session_state["generated"] = []  # AI responses
    if "is_loading" not in st.session_state:
        st.session_state["is_loading"] = False

    # Display the conversation history
    for user_msg, ai_msg in zip(st.session_state["past"], st.session_state["generated"]):
        st.markdown(f"**User:** {user_msg}")
        st.markdown(f"**AI:** {ai_msg}")

    # Get user input
    user_input = st.text_input("Your question about Streamlit:", "")

    # Optional file upload to append text content to the query
    uploaded_file = st.file_uploader(
        "Upload an Image/PDF, and its content will be appended to your prompt:",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf", "webp"],
    )

    # Placeholder for a loading spinner
    loading_placeholder = st.empty()

    if st.button("Send"):
        st.session_state["is_loading"] = True
        with loading_placeholder.container():
            st.write("Processing your request, please wait...")
            st.spinner("Loading...")

        # Combine the user query with any extracted text from an uploaded file
        combined_input = user_input.strip()
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    combined_input += f"\n\n[PDF TEXT EXTRACTED]:\n{pdf_text}"
            else:
                ocr_text = extract_text_from_image(uploaded_file)
                if ocr_text:
                    combined_input += f"\n\n[TEXT EXTRACTED FROM IMAGE]:\n{ocr_text}"

        try:
            # Here, "user123" is a placeholder session ID; replace it with a dynamic identifier as needed.
            answer = run_chat(combined_input, session_id="user123")
            st.success("Response generated successfully!")
        except Exception as e:
            traceback.print_exc()
            answer = f"Error: {e}"
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(answer)
        finally:
            st.session_state["is_loading"] = False
            loading_placeholder.empty()

        st.rerun()

if __name__ == "__main__":
    main()