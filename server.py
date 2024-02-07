import gradio as gr
from summarizer.text import get_text
from summarizer.openai_summarizer import questions_from_title, summarize_openai
from summarizer.args import Args


def summarize(title, url):
    args = Args(
        type="summary",
        display_prompt=False,
        no_generate=False,
        include_code=False,
        include_tables=False,
    )
    questions = questions_from_title(title)
    text = get_text(url, args)
    yield from summarize_openai(
        text,
        questions,
    )


# Define the Gradio interface
demo = gr.Interface(
    fn=summarize,  # The function to call
    inputs=["text", "text"],  # The input component
    outputs=["text"],  # The output component
)

# Launch the app
demo.launch()
