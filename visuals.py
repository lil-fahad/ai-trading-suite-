
import matplotlib.pyplot as plt
import streamlit as st

def plot_predictions(actual, predicted, title=""):
    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
