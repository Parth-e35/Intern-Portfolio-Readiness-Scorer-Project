import requests
import pandas as pd
import time
import os


OUTPUT_FILE = "data/extra_github_users.csv"


def get_headers(token=None):
    headers = {"Accept": "application/vnd.github+json"}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def search_github_users(query, category, token=None, pages=2,per_page=20):
    users = []

    for page in range(1, pages + 1):
        url = "https://api.github.com/search/users"

        params = {
            "q": query,
            "per_page": per_page,
            "page": page
        }

        response = requests.get(
            url,
            headers=get_headers(token),
            params=params,
            timeout=20
        )

        if response.status_code != 200:
            print("API Error:", response.status_code)
            print(response.text)
            break

        data = response.json()

        for item in data.get("items", []):
            username = item.get("login", "")
            profile_url = item.get("html_url", "")

            if username:
                users.append({
                    "candidate_name": f"Candidate_{username}",
                    "github_username": username,
                    "github_link": profile_url,
                    "expected_category": category,
                    "source": "GitHub Search API"
                })

        time.sleep(2)

    return users


def collect_users():
    os.makedirs("data", exist_ok=True)

    token = input("Enter GitHub token: ").strip()

    search_groups = {
        "Job Ready": [
            "repos:>10 followers:>10",
            "language:Python repos:>10 followers:>5",
            "language:JavaScript repos:>10 followers:>5",
            "language:Java repos:>10 followers:>5"
        ],

        "Almost Ready": [
            "repos:3..10 followers:1..10",
            "language:Python repos:3..8 followers:1..5",
            "language:JavaScript repos:3..8 followers:1..5",
            "language:HTML repos:3..8 followers:1..5"
        ],

        "Needs Improvement": [
            "repos:1..3 followers:0..2",
            "language:HTML repos:1..3 followers:0..2",
            "language:Python repos:1..3 followers:0..2",
            "language:JavaScript repos:1..3 followers:0..2"
        ]
    }

    all_users = []

    for category, queries in search_groups.items():
        print(f"\nCollecting profiles for category: {category}")

        for query in queries:
            print(f"Searching: {query}")

            users = search_github_users(
                query=query,
                category=category,
                token=token,
                pages=1,
                per_page=15
            )

            all_users.extend(users)

    df = pd.DataFrame(all_users)

    df = df.drop_duplicates(subset=["github_username"])

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\nCollection completed.")
    print(f"Total unique GitHub users collected: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nCategory distribution:")
    print(df["expected_category"].value_counts())


if __name__ == "__main__":
    collect_users()