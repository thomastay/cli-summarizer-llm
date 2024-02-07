# Like server.py, but you paste hacker news URLs here

import re
import gradio as gr
import requests
from summarizer.text import get_text
from summarizer.openai_summarizer import questions_from_title, summarize_openai
from summarizer.args import Args


def extract_id_from_hn_url(url):
    pattern = r"item\?id=(\d+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Invalid Hacker News URL: {url}")


def fetch_hn_story(id):
    url = f"https://hacker-news.firebaseio.com/v0/item/{id}.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            story = response.json()
            return story
        else:
            print(f"Failed to fetch story {id}. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request exception: {e}")
        return None


def extract_from_story(story):
    story_type = story["type"]
    if story_type != "story":
        print("Skipping story of type", story_type)
        return None
    url = story["url"]
    title = story["title"]
    return title, url


def summarize(url):
    args = Args(
        type="summary",
        display_prompt=False,
        no_generate=False,
        include_code=False,
        include_tables=False,
    )
    id = extract_id_from_hn_url(url)
    story = fetch_hn_story(id)
    if story is None:
        return "Failed to fetch story"
    extracted = extract_from_story(story)
    if extracted is None:
        return "Failed to extract story"
    title, article_url = extracted

    questions = questions_from_title(title)
    text = get_text(article_url, args)
    yield from summarize_openai(
        text,
        questions,
    )


# Define the Gradio interface
demo = gr.Interface(
    fn=summarize,  # The function to call
    inputs=["text"],  # The input component
    outputs=["text"],  # The output component
)

# Launch the app
demo.launch()
