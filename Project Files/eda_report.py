import os
import pandas as pd
import plotly.express as px


DATA_PATH = "data/final_portfolio_dataset.csv"
OUTPUT_FOLDER = "reports/eda_charts"


def create_eda():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("Dataset Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTarget Distribution:")
    print(df["readiness_category"].value_counts())

    fig1 = px.histogram(
        df,
        x="readiness_category",
        title="Readiness Category Distribution"
    )
    fig1.write_html(f"{OUTPUT_FOLDER}/readiness_distribution.html")

    fig2 = px.histogram(
        df,
        x="portfolio_score",
        color="readiness_category",
        title="Portfolio Score Distribution"
    )
    fig2.write_html(f"{OUTPUT_FOLDER}/portfolio_score_distribution.html")

    fig3 = px.scatter(
        df,
        x="total_repositories_analyzed",
        y="language_count",
        color="readiness_category",
        size="total_stars",
        title="Repositories vs Language Count"
    )
    fig3.write_html(f"{OUTPUT_FOLDER}/repo_vs_language.html")

    fig4 = px.scatter(
        df,
        x="readme_count",
        y="description_count",
        color="readiness_category",
        size="total_repositories_analyzed",
        title="Documentation vs Presentation"
    )
    fig4.write_html(f"{OUTPUT_FOLDER}/documentation_vs_presentation.html")

    numeric_cols = [
        "public_repos",
        "followers",
        "following",
        "total_repositories_analyzed",
        "total_stars",
        "total_forks",
        "total_watchers",
        "readme_count",
        "description_count",
        "homepage_count",
        "recent_repo_count",
        "language_count",
        "portfolio_score"
    ]

    corr = df[numeric_cols].corr()

    fig5 = px.imshow(
        corr,
        text_auto=True,
        title="Feature Correlation Heatmap"
    )
    fig5.write_html(f"{OUTPUT_FOLDER}/correlation_heatmap.html")

    print("\nEDA charts saved in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    create_eda()