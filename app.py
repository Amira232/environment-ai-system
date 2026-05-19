import gradio as gr
from transformers import pipeline
from PIL import Image, ImageDraw
import random

# Load Hugging Face Pipelines
classifier = pipeline(
    "sentiment-analysis"
)

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased"
)

# Sentiment Analysis

def analyze_text(text):

    if text == "":
        return "Please enter some text."

    result = classifier(text)[0]

    label = result["label"]
    score = round(result["score"] * 100, 2)

    return f"""
Prediction: {label}

Confidence: {score}%
"""

# AI Environmental Poster
def generate_environment_image(prompt):

    img = Image.new(
        "RGB",
        (500, 300),
        color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    )

    draw = ImageDraw.Draw(img)

    draw.text(
        (20, 120),
        f"🌍 {prompt}",
        fill="white"
    )

    return img

# Text Summarization
def summarize_text(text):

    if len(text.split()) < 30:
        return "Please enter longer text."

    summary = summarizer(
        text,
        max_length=60,
        min_length=20,
        do_sample=False
    )

    return summary[0]["summary_text"]

# Fill Mask Prediction
def predict_mask(text):

    if "[MASK]" not in text:
        return ["Please use [MASK] token."]

    results = fill_mask(text)

    output = []

    for r in results[:5]:
        output.append(r["sequence"])

    return output

# Gradio Interface
with gr.Blocks() as demo:

    gr.Markdown("""
    # 🌱 EcoVision AI

    Environmental NLP and AI Intelligence System
    """)

    with gr.Tab("🧠 Sentiment Analysis"):

        inp1 = gr.Textbox(
            label="Enter Environmental Text",
            lines=4
        )

        out1 = gr.Textbox(
            label="Analysis Result"
        )

        btn1 = gr.Button("Analyze")

        btn1.click(
            analyze_text,
            inp1,
            out1
        )

    with gr.Tab("🎨 AI Environmental Poster"):

        inp2 = gr.Textbox(
            label="Enter Poster Theme"
        )

        out2 = gr.Image(
            label="Generated Poster"
        )

        btn2 = gr.Button("Generate")

        btn2.click(
            generate_environment_image,
            inp2,
            out2
        )

    with gr.Tab("📝 Text Summarizer"):

        inp3 = gr.Textbox(
            label="Enter Long Environmental Article",
            lines=8
        )

        out3 = gr.Textbox(
            label="Summary"
        )

        btn3 = gr.Button("Summarize")

        btn3.click(
            summarize_text,
            inp3,
            out3
        )

    with gr.Tab("🧩 Fill Mask Prediction"):

        inp4 = gr.Textbox(
            label="Use [MASK] token"
        )

        out4 = gr.JSON(
            label="Predictions"
        )

        btn4 = gr.Button("Predict")

        btn4.click(
            predict_mask,
            inp4,
            out4
        )

demo.launch()
