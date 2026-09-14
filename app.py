import streamlit as st
import google.generativeai as genai
import PyPDF2
import pandas as pd
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 Multi-lingual AI Resume Screening Tool")
st.write("Upload a Job Description and up to 10 resumes. The AI supports English and multiple Indian languages.")

# 🚨 SECURE API CONFIGURATION 🚨
# The app now securely pulls the key directly from .streamlit/secrets.toml
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("API Key not found. Please ensure you have set up your .streamlit/secrets.toml file.")
    st.stop() # Stops the app from running further if the key is missing

# --- UTILITY FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def evaluate_resume(jd_text, resume_text, candidate_name):
    model = genai.GenerativeModel('gemini-3.6-flash')
    
    prompt = f"""
    You are an expert, unbiased AI technical recruiter. 
    Evaluate the candidate against the Job Description on Skills, Experience, and Culture.
    
    You MUST return the evaluation EXCLUSIVELY as a valid JSON object with the exact keys below.
    {{
        "Candidate": "{candidate_name}",
        "Skills_Score": <integer out of 10>,
        "Experience_Score": <integer out of 10>,
        "Culture_Score": <integer out of 10>,
        "Total_Score": <sum of the three scores, out of 30>,
        "Rationale": "<1-paragraph explanation of why they fit or fall short>",
        "Bias_Check": "<explicitly state if there are potential biases, or say 'No clear bias detected.'>"
    }}
    
    JOB DESCRIPTION:
    {jd_text}
    
    RESUME TEXT:
    {resume_text}
    """
    
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

# --- MAIN APP UI ---
st.subheader("1. Job Description")
jd_input = st.text_area("Paste the Job Description here:", height=150)

st.subheader("2. Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload 1 to 10 PDF resumes", type=["pdf"], accept_multiple_files=True)

if len(uploaded_files) > 10:
    st.error("⚠️ Please upload a maximum of 10 resumes at a time.")

if st.button("Analyze Resumes"):
    if not jd_input or len(uploaded_files) == 0 or len(uploaded_files) > 10:
        st.error("Please ensure you have a JD, at least 1 resume, and no more than 10.")
    else:
        progress_bar = st.progress(0)
        results = []
        
        for i, file in enumerate(uploaded_files):
            try:
                resume_text = extract_text_from_pdf(file)
                with st.spinner(f"Analyzing {file.name}..."):
                    eval_data = evaluate_resume(jd_input, resume_text, file.name)
                    results.append(eval_data)
            except Exception as e:
                st.error(f"Error processing {file.name}. Error: {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        st.success("Screening Complete!")
        
        if results:
            st.markdown("---")
            st.subheader("📊 Candidate Score Overview")
            
            df = pd.DataFrame(results)
            chart_data = df.set_index("Candidate")[["Total_Score"]]
            st.bar_chart(chart_data)
            
            st.subheader("📝 Detailed Analysis")
            sorted_results = sorted(results, key=lambda x: x["Total_Score"], reverse=True)
            
            for res in sorted_results:
                with st.expander(f"**{res['Candidate']}** — Total Score: {res['Total_Score']}/30"):
                    st.markdown(f"**Skills:** {res['Skills_Score']}/10 | **Experience:** {res['Experience_Score']}/10 | **Culture:** {res['Culture_Score']}/10")
                    st.markdown(f"**Rationale:** {res['Rationale']}")
                    st.info(f"**Bias Check:** {res['Bias_Check']}")