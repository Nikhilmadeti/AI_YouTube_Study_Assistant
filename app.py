import os
import gradio as gr
from dotenv import load_dotenv
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from urllib.parse import urlparse, parse_qs

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Please add GEMINI_API_KEY to your .env file.")

client = genai.Client(api_key=API_KEY)

transcript_text = ""


def extract_video_id(url):
    parsed = urlparse(url)

    if parsed.hostname == "youtu.be":
        return parsed.path[1:]

    if parsed.hostname in (
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
    ):
        return parse_qs(parsed.query)["v"][0]

    raise ValueError("Invalid YouTube URL")


def get_video_info(url):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "title": info.get("title"),
        "channel": info.get("uploader"),
        "duration": info.get("duration_string"),
    }


def get_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    return " ".join(
        snippet.text
        for snippet in transcript
    )


def summarize_video(url):
    global transcript_text

    try:
        video_id = extract_video_id(url)

        info = get_video_info(url)

        transcript_text = get_transcript(video_id)

        prompt = f"""
You are an expert study assistant.

Summarize this YouTube lecture.

Transcript:

{transcript_text}

Give:

1. Summary
2. Key Points
3. Important Definitions
4. Final Revision Notes
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        details = f"""
### Video Information

**Title:** {info['title']}

**Channel:** {info['channel']}

**Duration:** {info['duration']}
"""

        return details, response.text

    except Exception as e:
        return "", str(e)


def ask_question(question, history):

    global transcript_text

    if history is None:
        history = []

    if not transcript_text:
        history.append({
            "role": "assistant",
            "content": "Please generate a summary first."
        })
        return history, ""

    prompt = f"""
Answer ONLY using the transcript below.

Transcript:

{transcript_text}

Question:
{question}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        history = history + [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": response.text
            }
        ]

        return history, ""

    except Exception as e:

        history.append({
            "role": "assistant",
            "content": f"❌ {e}"
        })

        return history, ""


with gr.Blocks(title="AI YouTube Study Assistant") as demo:

    gr.Markdown("# 📺 AI YouTube Study Assistant")

    with gr.Tab("Summary"):

        url = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )

        info = gr.Markdown()

        summary = gr.Markdown()

        btn = gr.Button("Generate Summary")

        btn.click(
            summarize_video,
            inputs=url,
            outputs=[info, summary],
        )

    with gr.Tab("Chat"):

        chatbot = gr.Chatbot(height=500)

        question = gr.Textbox(
            placeholder="Ask something about the video..."
        )

        question.submit(
            ask_question,
            inputs=[question, chatbot],
            outputs=[chatbot, question],
        )

demo.launch()