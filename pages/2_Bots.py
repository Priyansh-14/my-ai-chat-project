import streamlit as st

def main():
    st.set_page_config(page_title="Available Bots", page_icon="🤖", layout="centered")

    # Check login status
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("You must be logged in to view available bots.")
        st.stop()

    st.title("Available Bots")
    st.markdown("---")

    # Example list of bots
    bots = [
        {
            "name": "StreamlitBot",
            "description": "Answers questions only about Streamlit.",
            "status": "Active",
            "icon": "🤖"
        },
        {
            "name": "WeatherBot",
            "description": "Provides weather updates (Coming Soon).",
            "status": "Inactive",
            "icon": "☁️"
        },
        {
            "name": "FinanceBot",
            "description": "Answers finance queries (Coming Soon).",
            "status": "Inactive",
            "icon": "💰"
        }
    ]

    cols = st.columns(len(bots))

    for i, bot in enumerate(bots):
        with cols[i]:
            st.subheader(f"{bot['icon']} {bot['name']}",anchor=f"{bot['name']}")
            st.write(f"**Description:** {bot['description']}")
            st.write(f"**Status:** `{bot['status']}`")
            # Example button if you want to direct to a specific page or function
            # if st.button(f"Open {bot['name']}"):
            #     st.write("Feature not implemented yet.")
            

def run():
    main()

if __name__ == "__main__":
    run()