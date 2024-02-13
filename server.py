import time
import gradio as gr
from summarizer.text import get_text_and_title
from summarizer.openai_summarizer import questions_from_title, summarize_openai
from summarizer.database import save_to_db


def summarize(url):
    title, text = get_text_and_title(url)
    questions = questions_from_title(title)

    final_output = ""
    for partial_out in summarize_openai(
        text,
        questions,
    ):
        final_output = partial_out
        yield partial_out
    key = url
    value = {
        "title": title,
        "text": text,
        "questions": questions,
        "summary": final_output,
        "created_at": round(time.time()),
    }
    save_to_db(key, value)


# Define the Gradio interface
demo = gr.Interface(
    fn=summarize,  # The function to call
    inputs=["text"],  # The input component
    outputs=["text"],  # The output component
)

# Launch the app
demo.launch()
