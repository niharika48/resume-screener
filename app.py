import streamlit as st
import google.generativeai as genai
import PyPDF2
import pandas as pd
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Resume Screener Pro", page_icon="💼", layout="wide")

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .candidate-card {
        border-left: 5px solid #1f77b4;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.4em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        background-color: #17a2b8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("💼 Multi-lingual AI Resume Screening Pro")
st.markdown("Automate your hiring process with blind screening, skill-gap analysis, and interview generation.")

# 🚨 SECURE API CONFIGURATION 🚨
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("API Key not found. Please ensure you have set up your Streamlit Secrets.")
    st.stop()

# --- UTILITY FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def evaluate_all_resumes(jd_text, resumes_dict):
    """Sends ONE request to Gemini containing all resumes with advanced prompt features."""
    model = genai.GenerativeModel('gemini-3.5-flash')
    
    prompt = f"""
    You are an expert, unbiased AI technical recruiter. 
    Below is a Job Description and the text from {len(resumes_dict)} candidates' resumes.
    
    CRITICAL INSTRUCTION FOR BLIND SCREENING: You must ignore candidates' names, gender, age, and location. Base your evaluation strictly on merit, skills, and experience.
    
    Evaluate EACH candidate against the Job Description on Skills, Experience, and Culture.
    
    You MUST return the evaluation EXCLUSIVELY as a valid JSON array containing one object per candidate.
    Use this exact structure for the JSON array:
    [
        {{
            "Candidate": "filename.pdf",
            "Language": "<Detected primary language of the resume>",
            "Skills_Score": <integer out of 10>,
            "Experience_Score": <integer out of 10>,
            "Culture_Score": <integer out of 10>,
            "Total_Score": <sum of the three scores, out of 30>,
            "Matched_Skills": ["skill1", "skill2"],
            "Missing_Skills": ["skill1", "skill2"],
            "Interview_Questions": ["Q1", "Q2", "Q3"],
            "Rationale": "<1-paragraph explanation of why they fit or fall short>",
            "Bias_Check": "<explicitly state if there are potential biases, or say 'No clear bias detected.'>"
        }}
    ]
    
    JOB DESCRIPTION:
    {jd_text}
    
    RESUMES TO EVALUATE:
    """
    
    for name, text in resumes_dict.items():
        prompt += f"\n\n--- START RESUME: {name} ---\n{text}\n--- END RESUME: {name} ---\n"
        
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

def convert_df_to_csv(df):
    # Select specific columns to export to keep the spreadsheet clean
    export_df = df.drop(columns=['Matched_Skills', 'Missing_Skills', 'Interview_Questions'], errors='ignore')
    return export_df.to_csv(index=False).encode('utf-8')

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("Upload Data")
    jd_input = st.text_area("Paste the Job Description here:", height=200)
    uploaded_files = st.file_uploader("Upload up to 10 PDF resumes", type=["pdf"], accept_multiple_files=True)
    
    if len(uploaded_files) > 10:
        st.error("⚠️ Please upload a maximum of 10 resumes.")
        
    analyze_btn = st.button("🚀 Analyze Candidates", use_container_width=True)

# --- MAIN LOGIC & UI ---
if analyze_btn:
    if not jd_input or len(uploaded_files) == 0 or len(uploaded_files) > 10:
        st.error("Please ensure you have a JD, at least 1 resume, and no more than 10.")
    else:
        with st.spinner("Analyzing resumes in a blind-screen batch..."):
            try:
                resumes_data = {file.name: extract_text_from_pdf(file) for file in uploaded_files}
                results = evaluate_all_resumes(jd_input, resumes_data)
                
                # Store results in session state to prevent losing data when clicking tabs/buttons
                st.session_state['results'] = results
                st.success("Screening Complete!")
            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")

# If we have results, display the dashboard
if 'results' in st.session_state:
    results = st.session_state['results']
    df = pd.DataFrame(results)
    
    # Create Tabs for a beautiful layout
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Ranks", "📝 Detailed Profiles", "⚖️ Candidate Matchup"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Candidate Rankings")
            # Score Threshold Slider
            min_score = st.slider("Minimum Total Score Filter", min_value=0, max_value=30, value=0)
            
            # Filter Data
            filtered_df = df[df['Total_Score'] >= min_score].sort_values(by="Total_Score", ascending=False)
            
            if not filtered_df.empty:
                # Prepare Multi-Dimension Chart
                chart_data = filtered_df.set_index("Candidate")[["Skills_Score", "Experience_Score", "Culture_Score"]]
                st.bar_chart(chart_data, height=400)
            else:
                st.warning("No candidates meet the minimum score threshold.")
        
        with col2:
            st.subheader("Export Results")
            st.markdown("Download a clean spreadsheet of the filtered candidates and their scores.")
            csv = convert_df_to_csv(filtered_df)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name='screened_candidates.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            st.markdown("---")
            st.metric(label="Total Candidates Screened", value=len(df))
            st.metric(label="Candidates Meeting Threshold", value=len(filtered_df))

    with tab2:
        st.subheader("In-Depth Analysis & Skill Gaps")
        sorted_results = sorted([r for r in results if r['Total_Score'] >= min_score], key=lambda x: x["Total_Score"], reverse=True)
        
        for res in sorted_results:
            st.markdown(f"""
            <div class="candidate-card">
                <h3>{res['Candidate']} <span class="badge">{res.get('Language', 'English')}</span></h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Score Metrics
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Score", f"{res['Total_Score']}/30")
            sc2.metric("Skills", f"{res['Skills_Score']}/10")
            sc3.metric("Experience", f"{res['Experience_Score']}/10")
            sc4.metric("Culture", f"{res['Culture_Score']}/10")
            
            st.markdown(f"**Rationale:** {res['Rationale']}")
            
            # Skill Gaps
            g1, g2 = st.columns(2)
            with g1:
                st.success("**✅ Matched Skills:**\n" + "\n".join([f"- {s}" for s in res.get('Matched_Skills', [])]))
            with g2:
                st.error("**❌ Missing Skills:**\n" + "\n".join([f"- {s}" for s in res.get('Missing_Skills', [])]))
            
            # Interview Questions
            with st.expander("🎯 Generated Interview Questions"):
                for idx, q in enumerate(res.get('Interview_Questions', [])):
                    st.markdown(f"**Q{idx+1}:** {q}")
            
            st.info(f"**Bias Check:** {res['Bias_Check']}")
            st.markdown("---")

    with tab3:
        st.subheader("Candidate Matchup")
        if len(filtered_df) >= 2:
            comp_col1, comp_col2 = st.columns(2)
            candidates = filtered_df['Candidate'].tolist()
            
            with comp_col1:
                cand1 = st.selectbox("Select Candidate A", options=candidates, index=0)
                c1_data = next(item for item in results if item["Candidate"] == cand1)
                st.metric("Total Score", c1_data['Total_Score'])
                st.markdown("**Matched Skills**")
                st.write(", ".join(c1_data.get('Matched_Skills', [])))
                st.markdown("**Rationale**")
                st.write(c1_data['Rationale'])
                
            with comp_col2:
                cand2 = st.selectbox("Select Candidate B", options=candidates, index=1)
                c2_data = next(item for item in results if item["Candidate"] == cand2)
                st.metric("Total Score", c2_data['Total_Score'])
                st.markdown("**Matched Skills**")
                st.write(", ".join(c2_data.get('Matched_Skills', [])))
                st.markdown("**Rationale**")
                st.write(c2_data['Rationale'])
        else:
            st.info("You need at least two candidates meeting the score threshold to use the comparison tool.")