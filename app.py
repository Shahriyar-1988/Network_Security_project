import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.pipeline.batch_prediction import run_batch_prediction

st.set_page_config(page_title="Network Security Batch Prediction", layout="centered")
st.title("Batch Prediction via File Path")
st.markdown("Enter the path to your input CSV file to run batch prediction using the trained model.")

# ---------- Input Path ----------
input_path = st.text_input(" Enter the full directory path to your input CSV file:")

# ---------- Run Prediction ----------
if input_path:
    if not os.path.isdir(input_path):
        st.error(" Invalid file path. Please make sure the file exists.")
    else:
        try:
            st.info(" Running batch prediction...")
            output_path = run_batch_prediction(input_path)
            st.success(" Prediction completed successfully!")

            # Show preview of predictions
            prediction_df = pd.read_csv(output_path)
            st.write("Prediction Preview:", prediction_df.head())

            # Show class distribution as a bar chart

            # Map class labels
            prediction_df['prediction_label'] = prediction_df['prediction'].map({0: 'Non_secure', 1: 'Secure'})

            # Count predictions
            class_counts = prediction_df['prediction_label'].value_counts()

            # Set colors
            colors = ['red' if label == 'Non_secure' else 'green' for label in class_counts.index]

            # Plot
            st.subheader(" Prediction Distribution")

            fig, ax = plt.subplots()
            bars = ax.bar(class_counts.index, class_counts.values, color=colors)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{height}', ha='center', va='bottom')

            ax.set_ylabel("Count")
            ax.set_title("Prediction: Secure vs Non_secure")
            st.pyplot(fig)


            # Download button
            with open(output_path, "rb") as f:
                st.download_button(
                    label=" Download Prediction CSV",
                    data=f,
                    file_name=os.path.basename(output_path),
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f" Error during prediction: {e}")
