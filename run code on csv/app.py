import streamlit as st
import pandas as pd
import os

from prompt import build_prompt
from llm import generate_code
from executor import execute_code

st.set_page_config(page_title="LLM Data Analyst", layout="wide")

st.title("📊 LLM-Powered Data Analyst")

st.markdown("Ask questions about your **sales CSV data**.")

# Load CSV preview
if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv", encoding="latin1")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
else:
    st.error("data.csv not found")

user_query = st.text_input(
    "Enter your analysis request:",
    placeholder="e.g. Show total sales per region as a bar chart"
)

if st.button("Run Analysis"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating code with LLM..."):
            prompt = build_prompt(user_query)
            code = generate_code(prompt)

        st.subheader("🧠 Generated Code")
        st.code(code, language="python")

        with st.spinner("Executing code..."):
            try:
                result = execute_code(code)

                if os.path.exists("output.png"):
                    st.subheader("📈 Visualization")
                    st.image("output.png")

                if result["result_df"] is not None:
                    st.subheader("📋 Result Table")
                    st.dataframe(result["result_df"])

                st.success("Analysis completed successfully.")

            except Exception as e:
                st.error(f"Execution failed: {e}")
