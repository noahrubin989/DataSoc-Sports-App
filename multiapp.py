import streamlit as st

class MultiApp:
    """Combining multiple streamlit applications."""
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application"""
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # app = st.sidebar.radio(
        app = st.selectbox(
            'Select which page you would like to go to:',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()