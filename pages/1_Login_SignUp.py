import streamlit as st

def main():
    st.set_page_config(page_title="Login / Sign Up", page_icon="🔒", layout="centered")
    st.title("🔒 Login / Sign Up")

    # Session state for auth
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = ""

    if not st.session_state["logged_in"]:
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            st.subheader("Welcome Back")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Log In", type="primary"):
                # Placeholder for real authentication
                if email and password:
                    st.session_state["logged_in"] = True
                    st.session_state["user_email"] = email
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid credentials. Please try again.")

        with tab_signup:
            st.subheader("Create Account")
            new_email = st.text_input("New Email", key="signup_email")
            new_password = st.text_input("New Password", type="password", key="signup_password")
            if st.button("Sign Up", type="primary"):
                # Placeholder for real sign-up logic
                if new_email and new_password:
                    st.success("Sign Up successful! Please log in.")
                else:
                    st.error("Please fill in all fields.")

    else:
        st.write(f"**You are logged in as:** {st.session_state['user_email']}")
        if st.button("Log Out"):
            st.session_state["logged_in"] = False
            st.session_state["user_email"] = ""
            st.success("You have logged out successfully.")

def run():
    main()

if __name__ == "__main__":
    run()