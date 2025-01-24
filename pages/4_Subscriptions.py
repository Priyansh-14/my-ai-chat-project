import streamlit as st

def main():
    st.set_page_config(page_title="Subscriptions", page_icon="💳", layout="centered")

    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("You need to log in to view subscriptions.")
        st.stop()

    st.title("Available Subscriptions")
    st.markdown("---")

    plans = [
        {
            "name": "Basic", 
            "price": "$5 / month", 
            "features": ["1 bot usage", "Basic support"], 
            "icon": "🔹"
        },
        {
            "name": "Pro", 
            "price": "$15 / month", 
            "features": ["3 bots usage", "Priority support"], 
            "icon": "💠"
        },
        {
            "name": "Enterprise", 
            "price": "$50 / month", 
            "features": ["Unlimited bots", "Dedicated support"], 
            "icon": "🔶"
        }
    ]

    cols = st.columns(len(plans))

    for i, plan in enumerate(plans):
        with cols[i]:
            st.subheader(f"{plan['icon']} {plan['name']}")
            st.write(f"**Price**: {plan['price']}")
            st.write("**Features:**")
            for f in plan["features"]:
                st.write(f"- {f}")
            if st.button(f"Subscribe to {plan['name']}"):
                st.success(f"Successfully subscribed to {plan['name']}!")

def run():
    main()

if __name__ == "__main__":
    run()