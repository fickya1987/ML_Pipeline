
import os
from groq import Groq
import streamlit as st

# Set your API key for Groq
GROQ_API_KEY = "your-api-key"

# Initialize the Groq client with the API key
client = Groq(api_key=GROQ_API_KEY)

# Predefined prompt message
predefined_prompt = """

Model,Accuracy,Precision,Recall,F1-score,ROC AUC
Logistic Regression,0.828125,0.8,0.6,0.6857142857142857,0.8568181818181818
K-Nearest Neighbors,0.640625,0.42105263157894735,0.4,0.41025641025641024,0.70625
Support Vector Machine,0.75,0.7,0.35,0.4666666666666667,0.8227272727272728
Decision Tree,0.65625,0.4375,0.35,0.3888888888888889,0.5727272727272728
AdaBoost,0.796875,0.7333333333333333,0.55,0.6285714285714286,0.8761363636363636
Gradient Boosting,0.71875,0.5714285714285714,0.4,0.47058823529411764,0.8363636363636364
Random Forest,0.765625,0.7272727272727273,0.4,0.5161290322580645,0.8039772727272727
Extra Trees,0.734375,0.6153846153846154,0.4,0.48484848484848486,0.765340909090909

which conclusions or insights u can find from this data?

"""

# Streamlit UI
st.title("Report for model evaluation")


st.write("### How Models Perform as per Model,Accuracy,Precision,Recall,F1-score,ROC AUC")


# Execute the chat completion request immediately
try:
    # Creating a chat completion request with the predefined prompt
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": predefined_prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    
    # Displaying the response from the Groq model
    response = chat_completion.choices[0].message.content
    st.write(response)
except Exception as e:
    st.error(f"An error occurred: {e}")
