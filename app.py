import streamlit as st
from multiapp import MultiApp
from apps import (explore_dataset, about)
import numpy as npa
import torch
import torch.nn as nn
import numpy as np

app = MultiApp()

st.markdown("""
            # Sports Analytics Dashboard App
            
            ### Created By: DataSoc Education Portfolio
            📊 [LinkedIn](https://www.linkedin.com/company/datasoc/)  
            
            """)

# Add all your applications here
# app.add_app("Home", home.app)
app.add_app("Explore this app", explore_dataset.app)
app.add_app("Learn about DataSoc", about.app)



# The main app
app.run()