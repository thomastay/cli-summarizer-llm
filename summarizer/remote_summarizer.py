import requests
import json

from .prompt import (
    summary_prompt_remote,
    qa_prompt_remote,
    bullet_prompt_remote,
    summary_params,
    bullet_params,
)
from .sentence_breakdown import format_sentences_str
from .timing import timing

repo_url = "https://github.com/thomastay/cli-summarizer-llm/"
title = "Summarizer"


@timing
def summarize_openrouter(
    text,
    args,  # Args from command line
    remote_args,  # Args relating to remote
    prompt_params,
):
    """
    Summarizes the given text using the OpenRouter API.

    Args:
        text (str): The input text to be summarized.
        args (Namespace): The arguments from the command line.
        remote_args (dict): The arguments relating to remote.
        prompt_params (dict): The parameters for the summary prompt.

    Raises:
        NotImplementedError: If the type of the summary is "topic".

    Returns:
        None
    """
    if args.type == "topic":
        raise NotImplementedError
    elif args.type == "qa":
        system, user = qa_prompt_remote(text)
    elif args.type == "bullet":
        formatted_sentences = format_sentences_str(text)
        # print(formatted_sentences)
        system, user = bullet_prompt_remote(formatted_sentences)
        prompt_params = bullet_params
    elif args.type == "summaryplus":
        summarize_openrouter_multi(
            text,
            args,
            remote_args,
            prompt_params,
        )
        return
    else:
        system, user = summary_prompt_remote(text)

    if args.display_prompt:
        # print(f"<im_start>system{system}<im_end>\n")
        # print(f"<im_start>user{user}<im_end>\n")
        # print(f"<im_start>assistant\n")
        print(f"{system}\n{user}\n")
    summary = openrouter_request(system, user, remote_args, prompt_params)
    print(summary)


def openrouter_request(system, user, remote_args, prompt_params):
    api_key = remote_args["api_key"]
    model_name = remote_args["model_name"]
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": repo_url,
            "X-Title": title,
        },
        data=json.dumps(
            {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": prompt_params["num_out"],
                "temperature": prompt_params["temperature"],
                "top_p": prompt_params["top_p"],
                "top_k": prompt_params["top_k"],
            }
        ),
    )
    data = response.json()
    try:
        summary = data["choices"][0]["message"]["content"]
        return summary
    except:
        with open("cli-summarizer-llm-response.json", "w") as f:
            json.dump(data, f, indent=4)


def summarize_openrouter_multi(
    text,
    args,
    remote_args,
    prompt_params,
):
    prompt_params = summary_params
    system, user = summary_prompt_remote(text)
    summary = openrouter_request(system, user, remote_args, prompt_params)
    if summary is None:
        return
    print(summary)
    print("=" * 80)
    prompt_params = bullet_params
    system, user = bullet_prompt_remote(text)
    qa = openrouter_request(system, user, remote_args, prompt_params)
    print(qa)
