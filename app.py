import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sales Predictor AI", page_icon="📈")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Loading the two pickle files you created
    with open('model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('scaler.pkl', 'rb') as s_file:
        scaler = pickle.load(s_file)
    return model, scaler

# --- MAIN APP ---
def main():
    st.title("🚀 Advertising Sales Predictor")
    st.markdown("Enter your marketing spend details below to predict total sales.")

    try:
        model, scaler = load_assets()
        
        # User Input Form
        with st.form("prediction_form"):
            st.subheader("Advertising Budget ($)")
            
            tv = st.number_input("TV Budget", min_value=0.0, value=150.0, step=1.0)
            radio = st.number_input("Radio Budget", min_value=0.0, value=25.0, step=1.0)
            newspaper = st.number_input("Newspaper Budget", min_value=0.0, value=15.0, step=1.0)
            
            submit_button = st.form_submit_button("Predict Now")

        if submit_button:
            # 1. Prepare data
            features = np.array([[tv, radio, newspaper]])
            
            # 2. Scale the data (using your 2nd pickle file)
            scaled_features = scaler.transform(features)
            
            # 3. Predict
            prediction = model.predict(scaled_features)
            
            # 4. Display Result
            st.success(f"### Predicted Sales: {round(float(prediction[0]), 2)} units")
            
            # Quick Visualization
            chart_data = pd.DataFrame({
                'Channel': ['TV', 'Radio', 'Newspaper'],
                'Spend': [tv, radio, newspaper]
            })
            st.bar_chart(chart_data.set_index('Channel'))

    except FileNotFoundError:
        st.error("Missing Files: Ensure 'model.pkl' and 'scaler.pkl' are in the same folder.")

if __name__ == "__main__":
    main()