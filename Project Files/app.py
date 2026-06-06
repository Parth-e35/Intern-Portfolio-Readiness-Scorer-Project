import os
import json
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from github_api import analyze_github_profile
from resume_extractor import process_single_resume


MODEL_PATH = "models/best_readiness_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
FEATURE_PATH = "models/model_features.json"
DATA_PATH = "data/final_portfolio_dataset.csv"


st.set_page_config(
    page_title="Intern Portfolio Readiness Scorer",
    page_icon="🚀",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e1b4b 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

.hero {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5);
    padding: 38px;
    border-radius: 26px;
    color: white;
    box-shadow: 0 22px 45px rgba(0,0,0,0.28);
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 12px;
    letter-spacing: -0.5px;
}

.hero p {
    font-size: 17px;
    color: #e0e7ff;
    line-height: 1.6;
}

div[data-testid="stTabs"] button {
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    padding: 14px 28px;
    border-radius: 14px 14px 0 0;
}

div[data-testid="stTabs"] [role="tablist"] {
    gap: 14px;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.09);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.14);
}

.success-box, .warning-box, .danger-box {
    padding: 18px;
    border-radius: 16px;
    font-weight: 600;
    margin-top: 12px;
}

.success-box {
    background: rgba(16,185,129,0.15);
    border: 1px solid #10b981;
    color: #d1fae5;
}

.warning-box {
    background: rgba(245,158,11,0.15);
    border: 1px solid #f59e0b;
    color: #fef3c7;
}

.danger-box {
    background: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    color: #fee2e2;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.14);
    margin-bottom: 16px;
    min-height: 130px;
}
.card h3 {
    color: white;
    margin-bottom: 10px;
}
.card p {
    color: #cbd5e1;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)


def get_github_token():
    try:
        return st.secrets["GITHUB_TOKEN"]
    except Exception:
        return os.getenv("GITHUB_TOKEN")


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    with open(FEATURE_PATH, "r") as file:
        features = json.load(file)

    return model, encoder, features

@st.cache_data(show_spinner=False, ttl=3600)
def cached_github_analysis(username, token):
    return analyze_github_profile(username, token=token)


@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def safe_divide(a, b):
    return 0 if b == 0 else a / b


def extract_username_from_github_link(link):
    if not link:
        return ""

    link = str(link).strip()
    link = link.replace("https://", "").replace("http://", "")
    link = link.replace("www.", "")
    link = link.split("?")[0].split("#")[0]

    parts = link.split("/")

    if len(parts) >= 2 and parts[0].lower() == "github.com":
        return parts[1].strip()

    return link.strip()


def create_ml_input(github_data):
    total_repos = github_data.get("total_repositories_analyzed", 0)

    return {
        "total_repositories_analyzed": total_repos,
        "language_count": github_data.get("language_count", 0),
        "readme_count": github_data.get("readme_count", 0),
        "description_count": github_data.get("description_count", 0),
        "homepage_count": github_data.get("homepage_count", 0),
        "recent_repo_count": github_data.get("recent_repo_count", 0),
        "total_stars": github_data.get("total_stars", 0),
        "total_forks": github_data.get("total_forks", 0),
        "followers": github_data.get("followers", 0),
        "readme_ratio": safe_divide(github_data.get("readme_count", 0), total_repos),
        "description_ratio": safe_divide(github_data.get("description_count", 0), total_repos),
        "recent_activity_ratio": safe_divide(github_data.get("recent_repo_count", 0), total_repos),
        "avg_stars_per_repo": safe_divide(github_data.get("total_stars", 0), total_repos)
    }


def calculate_scores(github_data):
    total_repos = github_data.get("total_repositories_analyzed", 0)
    languages = github_data.get("language_count", 0)
    readmes = github_data.get("readme_count", 0)
    descriptions = github_data.get("description_count", 0)
    homepages = github_data.get("homepage_count", 0)
    recent = github_data.get("recent_repo_count", 0)
    stars = github_data.get("total_stars", 0)
    forks = github_data.get("total_forks", 0)

    readme_ratio = safe_divide(readmes, total_repos)
    desc_ratio = safe_divide(descriptions, total_repos)
    recent_ratio = safe_divide(recent, total_repos)

    project_quality = min(10, total_repos * 1.5 + stars * 0.2 + forks * 0.3)
    technical = min(10, languages * 1.5)
    documentation = min(10, readme_ratio * 10)
    presentation = min(10, desc_ratio * 7 + homepages)
    business = min(10, 5 + stars * 0.3 + homepages * 0.5)
    consistency = min(10, recent_ratio * 10)

    portfolio_score = (
        project_quality * 0.20 +
        technical * 0.20 +
        documentation * 0.15 +
        presentation * 0.15 +
        business * 0.15 +
        consistency * 0.15
    ) * 10

    return {
        "Project Quality": round(project_quality, 2),
        "Technical Skills": round(technical, 2),
        "Documentation": round(documentation, 2),
        "Presentation": round(presentation, 2),
        "Business Impact": round(business, 2),
        "Consistency": round(consistency, 2),
        "Portfolio Score": round(portfolio_score, 2)
    }


def generate_recommendations(github_data, scores):
    recs = []

    if github_data.get("total_repositories_analyzed", 0) < 3:
        recs.append("Add at least 3–5 complete GitHub projects.")

    if github_data.get("language_count", 0) < 2:
        recs.append("Use more technologies such as Python, SQL, Power BI, Streamlit, React, or ML libraries.")

    if scores["Documentation"] < 6:
        recs.append("Improve README files with overview, features, screenshots, setup steps, and output explanation.")

    if scores["Presentation"] < 6:
        recs.append("Add repository descriptions, screenshots, and live demo links.")

    if scores["Consistency"] < 6:
        recs.append("Maintain recent GitHub activity and update projects regularly.")

    if scores["Business Impact"] < 6:
        recs.append("Build business-oriented projects that solve real-world problems.")

    if scores["Portfolio Score"] >= 80:
        recs.append("Portfolio is strong. Focus on deployment links, resume alignment, and interview preparation.")

    return recs


def predict_readiness(github_data):
    model, encoder, features = load_model_assets()

    ml_row = create_ml_input(github_data)
    input_df = pd.DataFrame([ml_row])[features]

    pred_encoded = model.predict(input_df)[0]
    prediction = encoder.inverse_transform([pred_encoded])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(input_df)[0].max() * 100

    return prediction, confidence, ml_row

def result_section(username, github_data):
    prediction, confidence, ml_row = predict_readiness(github_data)
    scores = calculate_scores(github_data)
    recs = generate_recommendations(github_data, scores)

    if prediction == "Job Ready":
        prediction_color = "#10b981"
        prediction_bg = "rgba(16,185,129,0.16)"
        message = "Strong portfolio signals detected. Candidate appears job-ready."
    elif prediction == "Almost Ready":
        prediction_color = "#f59e0b"
        prediction_bg = "rgba(245,158,11,0.16)"
        message = "Candidate is close to job-ready but needs some portfolio improvements."
    else:
        prediction_color = "#ef4444"
        prediction_bg = "rgba(239,68,68,0.16)"
        message = "Candidate needs portfolio improvement before being considered job-ready."

    st.markdown("## Prediction Result")

    prediction_html = f"""
    <div style="background:{prediction_bg}; border:2px solid {prediction_color}; padding:34px; border-radius:26px; text-align:center; margin:20px 0 28px 0;">
        <div style="font-size:15px; color:#cbd5e1; font-weight:700; letter-spacing:1.4px; margin-bottom:14px; font-family:'Inter', sans-serif;
    letter-spacing:-1px;">
            FINAL READINESS PREDICTION
        </div>
        <div style="font-size:58px; font-weight:800; color:{prediction_color}; line-height:1.1; font-family:'Inter', sans-serif;
    letter-spacing:-1px;">
            {prediction}
        </div>
        <div style="font-size:18px; color:#e2e8f0; margin-top:14px; font-family:'Inter', sans-serif;
    letter-spacing:-1px;">
            Model Confidence: {round(confidence, 2) if confidence else 0}%
        </div>
        <div style="font-size:16px; color:#cbd5e1; margin-top:14px; font-family:'Inter', sans-serif;
    letter-spacing:-1px;">
            {message}
        </div>
    </div>
    """

    st.components.v1.html(prediction_html, height=240)

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Portfolio Score", scores["Portfolio Score"])
    metric2.metric("Repositories", github_data.get("total_repositories_analyzed", 0))
    metric3.metric("Languages", github_data.get("language_count", 0))

    if confidence:
        st.progress(int(confidence))
        st.caption(f"Model confidence: {round(confidence, 2)}%")

    st.markdown("## GitHub Portfolio Summary")

    summary = pd.DataFrame([{
        "GitHub Username": username,
        "GitHub Name": github_data.get("github_name", ""),
        "Public Repos": github_data.get("public_repos", 0),
        "Followers": github_data.get("followers", 0),
        "Stars": github_data.get("total_stars", 0),
        "Forks": github_data.get("total_forks", 0),
        "README Count": github_data.get("readme_count", 0),
        "Description Count": github_data.get("description_count", 0),
        "Recent Repos": github_data.get("recent_repo_count", 0),
        "Languages": github_data.get("languages_used", "")
    }])

    st.dataframe(summary, use_container_width=True)

    st.markdown("## Score Breakdown")

    score_df = pd.DataFrame({
        "Criteria": ["Project Quality", "Technical Skills", "Documentation", "Presentation", "Business Impact", "Consistency"],
        "Score": [
            scores["Project Quality"],
            scores["Technical Skills"],
            scores["Documentation"],
            scores["Presentation"],
            scores["Business Impact"],
            scores["Consistency"]
        ]
    })

    fig = px.bar(
        score_df,
        x="Criteria",
        y="Score",
        text="Score",
        range_y=[0, 10],
        color="Score",
        color_continuous_scale="Blues",
        title="Portfolio Score Breakdown"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Personalized Recommendations")

    for rec in recs:
        st.write("- " + rec)

    with st.expander("View ML Input Features"):
        st.dataframe(pd.DataFrame([ml_row]), use_container_width=True)

st.markdown("""
<div class="hero">
    <h1>Intern Portfolio Readiness Scorer</h1>
    <p>
    A professional AI/ML system that evaluates intern portfolios using resume parsing,
    GitHub REST API feature extraction, machine learning prediction, and readiness analytics.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

tabs = st.tabs([
    "Predict Readiness",
    "Analytics Dashboard",
    "Dataset",
    "About Project"
])


with tabs[0]:
    st.markdown("## Analyze Candidate Portfolio")
    st.markdown("Use either GitHub profile link or resume upload. Both options are always available.")

    left_col, right_col = st.columns(2)

    github_username_from_link = ""
    github_username_from_resume = ""

    with left_col:
        st.markdown("""
        <div class="card">
            <h3>GitHub Portfolio Link</h3>
            <p>Enter GitHub profile link or username directly.</p>
        </div>
        """, unsafe_allow_html=True)

        github_link = st.text_input(
            "GitHub Profile",
            placeholder="https://github.com/username",
            label_visibility="collapsed"
        )

        if github_link:
            github_username_from_link = extract_username_from_github_link(github_link)

    with right_col:
        st.markdown("""
        <div class="card">
            <h3>Resume Upload</h3>
            <p>Upload PDF/DOCX resume for automatic GitHub extraction.</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".docx"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            extracted = process_single_resume(temp_path)

            #st.markdown("### Extracted Resume Information")
            #st.json(extracted)

            github_username_from_resume = extracted.get("github_username", "")

            if github_username_from_resume == "":
                st.warning("No GitHub link found in resume. Please use GitHub link input.")

    github_username = github_username_from_link or github_username_from_resume

    if st.button("Analyze Portfolio", use_container_width=True):
        if github_username == "":
            st.error("Please provide a valid GitHub profile link or upload a resume containing GitHub link.")
        else:
            token = get_github_token()

            with st.spinner("Fetching GitHub portfolio data..."):
                github_data = cached_github_analysis(github_username, token)

            if github_data is None:
                st.error("GitHub profile not found or API limit/error occurred.")
            else:
                result_section(github_username, github_data)
                
with tabs[1]:
    st.markdown("## Portfolio Analytics Dashboard")

    data = load_dataset()

    if data is None:
        st.warning("Dataset not found.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Candidates", len(data))
        c2.metric("Average Score", round(data["portfolio_score"].mean(), 2))
        c3.metric("Highest Score", round(data["portfolio_score"].max(), 2))
        c4.metric("Job Ready", len(data[data["readiness_category"] == "Job Ready"]))

        fig1 = px.pie(
            data,
            names="readiness_category",
            title="Readiness Category Distribution",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig1, use_container_width=True)

        top_data = data.sort_values("portfolio_score", ascending=False).head(20)

        category_avg = data.groupby("readiness_category")[["portfolio_score", "documentation_score", "technical_skills_score", "consistency_score"]].mean().reset_index()

        fig2 = px.bar(
            category_avg,
            x="readiness_category",
            y="portfolio_score",
            color="readiness_category",
            title="Average Portfolio Score by Readiness Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text_auto=True
        )
        st.plotly_chart(fig2, use_container_width=True)

        gap_columns = [
            "project_quality_score",
            "technical_skills_score",
            "documentation_score",
            "presentation_score",
            "business_impact_score",
            "consistency_score"
        ]

        gap_df = data[gap_columns].mean().reset_index()
        gap_df.columns = ["Evaluation Area", "Average Score"]

        fig3 = px.bar(
        gap_df,
        x="Evaluation Area",
        y="Average Score",
        title="Overall Portfolio Skill Gap Analysis",
        color="Average Score",
        color_continuous_scale="Blues",
        text_auto=True
        )
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter(
            data,
            x="readme_count",
            y="recent_repo_count",
            color="readiness_category",
            size="language_count",
            hover_name="github_username",
            title="Documentation Quality vs Recent GitHub Activity",
            color_discrete_sequence=px.colors.qualitative.Set2
                )
        st.plotly_chart(fig4, use_container_width=True)

        fig3 = px.scatter(
            data,
            x="total_repositories_analyzed",
            y="language_count",
            color="readiness_category",
            size="total_stars",
            hover_name="github_username",
            title="Repositories vs Technology Diversity",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig3, use_container_width=True)


with tabs[2]:
    st.markdown("## Final Dataset")

    data = load_dataset()

    if data is None:
        st.warning("Dataset not found.")
    else:
        st.dataframe(data, use_container_width=True)

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Dataset",
            data=csv,
            file_name="final_portfolio_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )


with tabs[3]:
    st.markdown("## About Project")

    st.markdown("""
    This project evaluates intern portfolio readiness using:

    - Resume parsing
    - GitHub link extraction
    - GitHub REST API
    - Feature engineering
    - Machine learning prediction
    - Dashboard analytics
    - Personalized recommendations

    The model uses GitHub-based features such as repositories, languages, README count,
    description count, recent activity, stars, forks and followers.
    """)