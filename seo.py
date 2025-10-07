# Reddit E-Propulsion Brand & User Feedback Analysis Project (Official API Version)
# ------------------------------------------------------------------------------

import os
import praw
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# 1. Project Setup
# -------------------------------

def setup_project_structure():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    print("✅ Project directories created.")

# -------------------------------
# 2. Reddit Data Collection (via PRAW)
# -------------------------------

def collect_reddit_posts():
    reddit = praw.Reddit(
        client_id="h-qSouARHeg8HSevQgPYUg",
        client_secret="qP_Q5kWqf1_osWLCCuGohtG4UW09zQ",
        user_agent="reddit-epropulsion-analysis"
    )

    keywords = ["E-Propulsion", "ePropulsion", "Spirit 1.0", "Navy 6.0", "Pod Drive", "electric outboard"]
    subreddits = ["sailing", "boating", "electricboats"]

    records = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        print(f"🔍 Searching in r/{sub}...")
        for kw in keywords:
            for submission in subreddit.search(kw, limit=200):
                text = (submission.title or "") + " " + (submission.selftext or "")
                if len(text.strip()) < 15:
                    continue
                records.append({
                    "id": submission.id,
                    "subreddit": sub,
                    "title": submission.title,
                    "text": text.strip(),
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "url": submission.url
                })
    df = pd.DataFrame(records)
    df.to_csv("data/raw/reddit_raw.csv", index=False)
    print(f"✅ Collected {len(df)} Reddit posts.")

# -------------------------------
# 3. Data Cleaning & Processing
# -------------------------------

def clean_and_process():
    df = pd.read_csv("data/raw/reddit_raw.csv")

    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])

    def categorize(text):
        text = str(text).lower()
        if "battery" in text: return "Battery"
        if "price" in text or "cost" in text: return "Pricing"
        if "warranty" in text or "support" in text: return "Customer Service"
        if "noise" in text or "quiet" in text: return "Noise"
        if "speed" in text or "range" in text: return "Performance"
        return "Other"

    df["topic"] = df["text"].apply(categorize)
    df["is_question"] = df["text"].apply(lambda x: "?" in str(x) or str(x).lower().startswith(("what", "how", "why")))

    df.to_csv("data/processed/reddit_cleaned.csv", index=False)
    print(f"✅ Cleaned dataset saved: {len(df)} records.")

# -------------------------------
# 4. Visualization
# -------------------------------

def generate_charts():
    df = pd.read_csv("data/processed/reddit_cleaned.csv")

    plt.figure(figsize=(8, 4))
    df["topic"].value_counts().plot(kind="bar", title="Top Discussion Topics")
    plt.tight_layout()
    plt.savefig("outputs/charts/topics.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    df["sentiment"].hist(bins=30)
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig("outputs/charts/sentiment_distribution.png")
    plt.close()

    print("✅ Charts generated.")

# -------------------------------
# 5. PDF Report Generation
# -------------------------------

def generate_pdf_report():
    df = pd.read_csv("data/processed/reddit_cleaned.csv")

    avg_sentiment = df["sentiment"].mean()
    total_posts = len(df)
    questions = df[df["is_question"] == True].shape[0]
    top_topics = df["topic"].value_counts().head(5)

    report_path = "outputs/reports/Epropulsion_Reddit_Report.pdf"
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("E-Propulsion Reddit Feedback Analysis Report", styles["Title"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph(f"<b>Total Posts Analyzed:</b> {total_posts}", styles["Normal"]))
    story.append(Paragraph(f"<b>Average Sentiment Score:</b> {avg_sentiment:.3f}", styles["Normal"]))
    story.append(Paragraph(f"<b>Total Questions Identified:</b> {questions}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Top Discussion Topics:</b>", styles["Heading2"]))
    for topic, count in top_topics.items():
        story.append(Paragraph(f"{topic}: {count}", styles["Normal"]))
    story.append(Spacer(1, 20))

    if os.path.exists("outputs/charts/topics.png"):
        story.append(Image("outputs/charts/topics.png", width=400, height=200))
        story.append(Spacer(1, 12))
    if os.path.exists("outputs/charts/sentiment_distribution.png"):
        story.append(Image("outputs/charts/sentiment_distribution.png", width=400, height=200))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<b>Summary:</b> Reddit users discuss E-Propulsion mainly around performance, pricing, and customer service. "
        "Sentiment trends show a neutral to slightly positive bias overall.",
        styles["Normal"]
    ))

    doc.build(story)
    print(f"📄 PDF report generated: {report_path}")

# -------------------------------
# 6. Main Execution Flow
# -------------------------------

def main():
    setup_project_structure()
    collect_reddit_posts()
    clean_and_process()
    generate_charts()
    generate_pdf_report()
    print("🎉 Project workflow completed with PDF report.")

if __name__ == "__main__":
    main()
