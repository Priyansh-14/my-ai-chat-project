import streamlit as st

def main():
    st.set_page_config(page_title="My Profile", page_icon="👤", layout="centered")

    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("Please log in to view your profile.")
        st.stop()

    st.title("My Profile")
    st.markdown("---")

    st.subheader(f"Welcome, {st.session_state['user_email']}!")
    st.write("**Name**: John Doe")
    st.write("**Membership Tier**: Premium")
    st.write("**Joined**: January 1, 2025")

    st.markdown("### Subscriptions Purchased")
    st.write("- Advanced Bot Subscription")
    st.write("- Monthly Chatbot Pack")

    if st.button("Log Out"):
        st.session_state["logged_in"] = False
        st.session_state["user_email"] = ""
        st.success("Logged out successfully!")

def run():
    main()

if __name__ == "__main__":
    run()