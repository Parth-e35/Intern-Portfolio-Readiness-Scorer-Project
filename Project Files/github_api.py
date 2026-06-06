import requests
from datetime import datetime


BASE_URL = "https://api.github.com"


def get_headers(token=None):
    headers = {
        "Accept": "application/vnd.github+json"
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def get_user_profile(username, token=None):
    url = f"{BASE_URL}/users/{username}"

    try:
        response = requests.get(url, headers=get_headers(token), timeout=15)

        if response.status_code != 200:
            return None

        return response.json()

    except Exception:
        return None


def get_user_repositories(username, token=None):
    url = f"{BASE_URL}/users/{username}/repos?per_page=100&sort=updated"

    try:
        response = requests.get(url, headers=get_headers(token), timeout=20)

        if response.status_code != 200:
            print(f"Repo API error for {username}: {response.status_code}")
            print(response.text)
            return []

        data = response.json()

        if not isinstance(data, list):
            print(f"Unexpected repo data for {username}: {data}")
            return []

        return data

    except Exception as e:
        print(f"Repository fetch error for {username}: {e}")
        return []

def get_repo_languages(username, repo_name, token=None):
    url = f"{BASE_URL}/repos/{username}/{repo_name}/languages"

    try:
        response = requests.get(url, headers=get_headers(token), timeout=15)

        if response.status_code != 200:
            return []

        return list(response.json().keys())

    except Exception:
        return []


def check_readme_exists(username, repo_name, token=None):
    url = f"{BASE_URL}/repos/{username}/{repo_name}/readme"

    try:
        response = requests.get(url, headers=get_headers(token), timeout=15)

        if response.status_code == 200:
            return 1

        return 0

    except Exception:
        return 0


def days_since_updated(updated_at):
    try:
        updated_date = datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ")
        today = datetime.utcnow()
        return (today - updated_date).days

    except Exception:
        return 9999


def analyze_github_profile(username, token=None):
    profile = get_user_profile(username, token)

    if profile is None:
        return None

    repos = get_user_repositories(username, token)

    total_repos = len(repos)
    total_stars = 0
    total_forks = 0
    total_watchers = 0
    readme_count = 0
    description_count = 0
    homepage_count = 0
    recent_repo_count = 0
    languages = set()

    MAX_REPOS_TO_ANALYZE = 30
    
    repos = sorted(
    repos,
    key=lambda x: x.get("updated_at", ""),
    reverse=True
)

    repos = repos[:MAX_REPOS_TO_ANALYZE]

    for repo in repos:
        if not isinstance(repo, dict):
            print(f"Skipping invalid repo data for {username}: {repo}")
            continue

        repo_name = repo.get("name", "")

        if repo_name == "":
            continue

        total_stars += repo.get("stargazers_count", 0)
        total_forks += repo.get("forks_count", 0)
        total_watchers += repo.get("watchers_count", 0)

        if repo.get("description"):
            description_count += 1

        if repo.get("homepage"):
            homepage_count += 1

        readme_count += check_readme_exists(username, repo_name, token)

        repo_languages = get_repo_languages(username, repo_name, token)

        for language in repo_languages:
            languages.add(language)

        updated_at = repo.get("updated_at", "")
        if days_since_updated(updated_at) <= 180:
            recent_repo_count += 1

    github_data = {
        "github_username": username,
        "github_name": profile.get("name", ""),
        "github_bio": profile.get("bio", ""),
        "public_repos": profile.get("public_repos", 0),
        "followers": profile.get("followers", 0),
        "following": profile.get("following", 0),
        "total_repositories_analyzed": total_repos,
        "total_stars": total_stars,
        "total_forks": total_forks,
        "total_watchers": total_watchers,
        "readme_count": readme_count,
        "description_count": description_count,
        "homepage_count": homepage_count,
        "recent_repo_count": recent_repo_count,
        "languages_used": ", ".join(sorted(languages)),
        "language_count": len(languages)
    }

    return github_data