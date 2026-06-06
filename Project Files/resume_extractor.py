import os
import re
import pandas as pd
import PyPDF2
import docx


RESUME_FOLDER = "resumes"
OUTPUT_FILE = "data/resume_extracted_data.csv"
FAILED_FILE = "data/failed_resumes.csv"


def extract_text_from_pdf(file_path):
    text = ""

    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"PDF extraction error in {file_path}: {e}")

    return text


def extract_text_from_docx(file_path):
    text = ""

    try:
        document = docx.Document(file_path)

        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

    except Exception as e:
        print(f"DOCX extraction error in {file_path}: {e}")

    return text


def extract_text_from_resume(file_path):
    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_path)

    elif file_path_lower.endswith(".docx"):
        return extract_text_from_docx(file_path)

    return ""


def extract_email(text):
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = re.findall(pattern, text)

    if emails:
        return emails[0]

    return ""


def extract_phone(text):
    pattern = r"(\+91[\-\s]?)?[6-9]\d{9}"
    match = re.search(pattern, text)

    if match:
        return match.group()

    return ""


def extract_links(text):
    pattern = r"https?://[^\s\)\]\}>,]+|www\.[^\s\)\]\}>,]+"
    links = re.findall(pattern, text)

    cleaned_links = []

    for link in links:
        link = link.strip()
        link = link.rstrip(".,;)")
        cleaned_links.append(link)

    return list(set(cleaned_links))


def classify_links(links):
    github_link = ""
    linkedin_link = ""
    portfolio_link = ""

    for link in links:
        lower_link = link.lower()

        if "github.com" in lower_link and github_link == "":
            github_link = link

        elif "linkedin.com" in lower_link and linkedin_link == "":
            linkedin_link = link

        elif any(keyword in lower_link for keyword in [
            "vercel.app",
            "netlify.app",
            "github.io",
            "portfolio",
            "web.app",
            "firebaseapp.com"
        ]) and portfolio_link == "":
            portfolio_link = link

    return github_link, linkedin_link, portfolio_link


def extract_github_username(github_link):
    if github_link == "":
        return ""

    link = github_link.replace("https://", "").replace("http://", "")
    link = link.replace("www.", "")

    parts = link.split("/")

    if len(parts) >= 2 and parts[0].lower() == "github.com":
        username = parts[1].strip()

        invalid_words = ["", "features", "topics", "collections", "explore", "login"]

        if username not in invalid_words:
            return username

    return ""


def guess_candidate_name(file_name):
    name = os.path.splitext(file_name)[0]
    name = name.replace("_", " ").replace("-", " ")
    return name.title()


def process_single_resume(file_path):
    file_name = os.path.basename(file_path)

    text = extract_text_from_resume(file_path)
    links = extract_links(text)

    github_link, linkedin_link, portfolio_link = classify_links(links)
    github_username = extract_github_username(github_link)

    return {
        "candidate_name": guess_candidate_name(file_name),
        "resume_file": file_name,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "github_link": github_link,
        "github_username": github_username,
        "linkedin_link": linkedin_link,
        "portfolio_link": portfolio_link
    }


def process_all_resumes():
    os.makedirs("data", exist_ok=True)

    extracted_rows = []
    failed_rows = []

    if not os.path.exists(RESUME_FOLDER):
        os.makedirs(RESUME_FOLDER)
        print("resumes folder created. Add resumes and run again.")
        return

    for file_name in os.listdir(RESUME_FOLDER):
        if not file_name.lower().endswith((".pdf", ".docx")):
            continue

        file_path = os.path.join(RESUME_FOLDER, file_name)

        print(f"Processing: {file_name}")

        data = process_single_resume(file_path)

        if data["github_username"] == "":
            failed_rows.append({
                "resume_file": file_name,
                "reason": "GitHub link not found or invalid"
            })
        else:
            extracted_rows.append(data)

    extracted_df = pd.DataFrame(extracted_rows)
    failed_df = pd.DataFrame(failed_rows)

    extracted_df.to_csv(OUTPUT_FILE, index=False,encoding="utf-8")
    failed_df.to_csv(FAILED_FILE, index=False,encoding="utf-8")

    print("\nResume extraction completed.")
    print(f"Valid extracted resumes: {len(extracted_df)}")
    print(f"Failed resumes: {len(failed_df)}")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {FAILED_FILE}")


if __name__ == "__main__":
    process_all_resumes()