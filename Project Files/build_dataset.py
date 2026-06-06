import os
import pandas as pd

from github_api import analyze_github_profile
from recommendation_engine import generate_recommendations


INPUT_FILE = "data/resume_extracted_data.csv"
GITHUB_OUTPUT_FILE = "data/github_portfolio_data.csv"
FINAL_OUTPUT_FILE = "data/final_portfolio_dataset.csv"
FAILED_GITHUB_FILE = "data/failed_github_profiles.csv"


def safe_divide(a, b):
    return 0 if b == 0 else a / b


def extract_username_from_link(github_link):
    if pd.isna(github_link) or str(github_link).strip() == "":
        return ""

    link = str(github_link).strip()
    link = link.replace("https://", "").replace("http://", "")
    link = link.replace("www.", "")
    link = link.split("?")[0].split("#")[0]

    parts = link.split("/")

    if len(parts) >= 2 and parts[0].lower() == "github.com":
        return parts[1].strip()

    return ""


def calculate_scores(row):
    total_repos = row["total_repositories_analyzed"]
    language_count = row["language_count"]
    readme_count = row["readme_count"]
    description_count = row["description_count"]
    homepage_count = row["homepage_count"]
    recent_repo_count = row["recent_repo_count"]
    total_stars = row["total_stars"]
    total_forks = row["total_forks"]

    readme_ratio = safe_divide(readme_count, total_repos)
    description_ratio = safe_divide(description_count, total_repos)
    recent_activity_ratio = safe_divide(recent_repo_count, total_repos)

    project_quality_score = min(10, total_repos * 1.5 + total_stars * 0.2 + total_forks * 0.3)
    technical_skills_score = min(10, language_count * 1.5)
    documentation_score = min(10, readme_ratio * 10)
    presentation_score = min(10, description_ratio * 7 + homepage_count * 1)
    business_impact_score = min(10, 5 + total_stars * 0.3 + homepage_count * 0.5)
    consistency_score = min(10, recent_activity_ratio * 10)

    portfolio_score = (
        project_quality_score * 0.20 +
        technical_skills_score * 0.20 +
        documentation_score * 0.15 +
        presentation_score * 0.15 +
        business_impact_score * 0.15 +
        consistency_score * 0.15
    ) * 10

    if portfolio_score >= 80:
        readiness_category = "Job Ready"
    elif portfolio_score >= 60:
        readiness_category = "Almost Ready"
    else:
        readiness_category = "Needs Improvement"

    return {
        "project_quality_score": round(project_quality_score, 2),
        "technical_skills_score": round(technical_skills_score, 2),
        "documentation_score": round(documentation_score, 2),
        "presentation_score": round(presentation_score, 2),
        "business_impact_score": round(business_impact_score, 2),
        "consistency_score": round(consistency_score, 2),
        "portfolio_score": round(portfolio_score, 2),
        "readiness_category": readiness_category
    }


def build_dataset():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print("resume_extracted_data.csv not found.")
        print("Run: python resume_extractor.py first")
        return

    try:
        resume_df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        resume_df = pd.read_csv(INPUT_FILE, encoding="cp1252")

    if resume_df.empty:
        print("No valid resumes found.")
        return

    github_rows = []
    final_rows = []
    failed_rows = []

    token = input("Enter GitHub token optional, press Enter to skip: ").strip()
    token = token if token else None

    for index, resume_row in resume_df.iterrows():
        github_username = str(resume_row.get("github_username", "")).strip()
        github_link = str(resume_row.get("github_link", "")).strip()

        if github_username == "" or github_username.lower() == "nan":
            github_username = extract_username_from_link(github_link)

        username = github_username.strip()

        if username == "" or username.lower() == "nan":
            failed_rows.append({
                "candidate_name": resume_row.get("candidate_name", ""),
                "github_link": github_link,
                "github_username": "",
                "reason": "GitHub username/link missing"
            })
            continue

        print(f"\nCollecting GitHub data for: {username}")

        github_data = analyze_github_profile(username, token)

        if github_data is None:
            failed_rows.append({
                "candidate_name": resume_row.get("candidate_name", ""),
                "github_link": github_link,
                "github_username": username,
                "reason": "GitHub profile not found or API error"
            })
            continue

        github_rows.append(github_data)

        combined_row = {}
        combined_row.update(resume_row.to_dict())
        combined_row.update(github_data)

        scores = calculate_scores(combined_row)
        combined_row.update(scores)

        combined_row["recommendations"] = generate_recommendations(combined_row)

        final_rows.append(combined_row)

    github_df = pd.DataFrame(github_rows)
    final_df = pd.DataFrame(final_rows)
    failed_df = pd.DataFrame(failed_rows)

    github_df.to_csv(GITHUB_OUTPUT_FILE, index=False, encoding="utf-8")
    final_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding="utf-8")
    failed_df.to_csv(FAILED_GITHUB_FILE, index=False, encoding="utf-8")

    print("\nDataset building completed.")
    print(f"Total successful profiles: {len(final_df)}")
    print(f"Total failed profiles: {len(failed_df)}")
    print(f"GitHub data saved: {GITHUB_OUTPUT_FILE}")
    print(f"Final dataset saved: {FINAL_OUTPUT_FILE}")
    print(f"Failed GitHub profiles saved: {FAILED_GITHUB_FILE}")


if __name__ == "__main__":
    build_dataset()