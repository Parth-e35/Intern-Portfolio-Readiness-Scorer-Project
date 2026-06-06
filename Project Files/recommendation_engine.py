def generate_recommendations(row):
    recommendations = []

    if row["total_repositories_analyzed"] < 3:
        recommendations.append("Add at least 3 to 5 complete projects on GitHub.")

    if row["language_count"] < 2:
        recommendations.append("Use more technologies such as Python, SQL, Power BI, Streamlit, React, or ML libraries.")

    if row["documentation_score"] < 6:
        recommendations.append("Improve README files with overview, features, setup steps, screenshots, and output explanation.")

    if row["presentation_score"] < 6:
        recommendations.append("Add proper repository descriptions and live demo or portfolio links.")

    if row["consistency_score"] < 6:
        recommendations.append("Update GitHub regularly and maintain recent project activity.")

    if row["business_impact_score"] < 6:
        recommendations.append("Build projects that solve real business problems.")

    if row["portfolio_score"] >= 80:
        recommendations.append("Portfolio is strong. Focus on deployment links, resume alignment, and interview preparation.")

    return " | ".join(recommendations)