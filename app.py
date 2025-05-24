import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("GenAI Medical Diagnosis App")
st.subheader("Enter your symptoms below:")

symptoms = st.text_area("Symptoms", height=150)

if st.button("Get Diagnosis"):
    if symptoms.strip() == "":
        st.warning("Please enter your symptoms.")
    else:
        with st.spinner("Analyzing symptoms..."):
            prompt = f"You are a medical assistant. A patient reports the following symptoms: {symptoms}. What could be the most likely diagnosis?"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=100)
            diagnosis = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success(" Diagnosis:")
            st.markdown(f"**{diagnosis.strip()}**")

st.markdown("---")
st.caption("This is a GenAI demo and not a substitute for professional medical advice.")
