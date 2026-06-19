import os
import json
import traceback
from datetime import datetime
import pandas as pd
import plotly.express as px
import gradio as gr

try:
    from groq import Groq
    client = Groq(api_key="your_api_key_here")
except Exception:
    client = None

MOOD_CATEGORIES = ["Happy", "Sad", "Angry", "Relaxed", "Stressed"]
DATA_PATH = "mood_data.csv"


def load_data():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce").fillna(0).astype(int)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Timestamp", "Description", "Mood", "Confidence"])


def save_data(df):
    df_copy = df.copy()
    df_copy["Timestamp"] = df_copy["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_copy.to_csv(DATA_PATH, index=False)


DF = load_data()


def analyze_mood_groq(text):
    if client is None:
        raise RuntimeError("Groq client not initialized")

    prompt = (
        "You are an emotion analysis model. "
        "Classify the mood of the following text strictly as one of these categories: "
        "['Happy', 'Sad', 'Angry', 'Relaxed', 'Stressed'].\n\n"
        "Also estimate a confidence level (0-100).\n"
        "Return only JSON like this:\n"
        "{\"mood\": \"Sad\", \"confidence\": 92}\n\n"
        f"User text: {text}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are an emotion classification assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    try:
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        json_str = raw[json_start:json_end]
        parsed = json.loads(json_str)
        mood = parsed.get("mood", "").title()
        confidence = int(parsed.get("confidence", 70))
    except Exception:
        mood, confidence = analyze_mood_local(text)
        return mood, confidence

    if mood not in MOOD_CATEGORIES:
        mood = "Relaxed"
    return mood, min(max(confidence, 0), 100)


def analyze_mood_local(text):
    text = text.lower()
    keywords = {
        "Happy": ["happy", "joy", "smile", "glad", "excited", "great", "won", "success"],
        "Sad": ["sad", "unhappy", "cry", "lonely", "depressed", "down", "lose", "lost"],
        "Angry": ["angry", "mad", "furious", "irritated", "hate", "annoyed", "rage"],
        "Relaxed": ["relaxed", "calm", "peace", "chill", "rested", "content"],
        "Stressed": ["stressed", "worried", "tired", "panic", "pressure", "anxious", "nervous"],
    }

    scores = {m: 0 for m in MOOD_CATEGORIES}
    for mood, words in keywords.items():
        for w in words:
            if w in text:
                scores[mood] += 1
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    conf = int(min(95, (scores[best] / (total + 1)) * 100)) if total > 0 else 40
    return best, conf


def analyze_mood(text):
    if client:
        try:
            return analyze_mood_groq(text)
        except Exception:
            traceback.print_exc()
    return analyze_mood_local(text)


def add_entry(desc):
    global DF
    mood, conf = analyze_mood(desc)
    new_row = {
        "Timestamp": datetime.now(),
        "Description": desc,
        "Mood": mood,
        "Confidence": conf,
    }
    DF = pd.concat([DF, pd.DataFrame([new_row])], ignore_index=True)
    save_data(DF)
    return f"Detected: {mood} ({conf}% confidence)", DF.tail(20)


def generate_dashboard(date_text=None):
    try:
        df = load_data()
        if df.empty:
            return "No entries yet.", None, None

        if date_text and date_text.strip():
            try:
                date_obj = datetime.strptime(date_text.strip(), "%Y-%m-%d").date()
                df = df[df["Timestamp"].dt.date == date_obj]
            except Exception:
                return "Invalid date format. Use YYYY-MM-DD.", None, None

        if df.empty:
            return "No data for this date.", None, None

        counts = df["Mood"].value_counts().reindex(MOOD_CATEGORIES, fill_value=0)
        pie = px.pie(names=counts.index, values=counts.values, title="Mood Distribution")

        df_sorted = df.sort_values("Timestamp")
        line = px.line(
            df_sorted, x="Timestamp", y="Confidence", color="Mood",
            title="Confidence Over Time", markers=True
        )
        line.update_yaxes(range=[0, 100])

        top = counts.idxmax()
        summary = f"Most frequent mood: {top} ({counts[top]} times)"
        return summary, pie, line

    except Exception as e:
        traceback.print_exc()
        return f"Error generating dashboard: {str(e)}", None, None


def create_ui():
    with gr.Blocks(title="Mood Tracker") as demo:
        gr.Markdown("## Mood Tracker\nDescribe how you feel, get mood and confidence, then view analytics.")

        with gr.Tab("Log Mood"):
            txt = gr.Textbox(lines=3, placeholder="Describe your mood or how your day feels...")
            btn = gr.Button("Analyze & Save")
            status = gr.Textbox(label="Detected Mood", interactive=False)
            table = gr.Dataframe(value=DF.tail(20), label="Recent Entries")

            def on_submit(text):
                return add_entry(text)

            btn.click(on_submit, inputs=txt, outputs=[status, table])

        with gr.Tab("Analytics"):
            date_input = gr.Textbox(label="Filter by Date (YYYY-MM-DD, optional)")
            dash_btn = gr.Button("Generate Dashboard")
            summary = gr.Textbox(label="Summary", interactive=False)
            pie_plot = gr.Plot(label="Mood Distribution")
            line_plot = gr.Plot(label="Confidence Over Time")

            dash_btn.click(
                fn=generate_dashboard,
                inputs=date_input,
                outputs=[summary, pie_plot, line_plot],
            )

    return demo


if __name__ == "__main__":
    app = create_ui()
    app.launch()
