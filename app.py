import streamlit as st
from multiapp import MultiApp
from apps import (explore_dataset, about)
import numpy as npa
import torch
import torch.nn as nn
import numpy as np

app = MultiApp()

st.markdown("""
            # Finance App
            
            ### Created By: Noah Rubin
            📊 [LinkedIn](https://www.linkedin.com/in/noah-rubin1/)  
            
            🧑🏽‍💻 [GitHub](https://github.com/noahrubin989)
            """)

# Add all your applications here
# app.add_app("Home", home.app)
app.add_app("Explore this app", explore_dataset.app)
app.add_app("Learn about me", about.app)



# The main app
app.run()