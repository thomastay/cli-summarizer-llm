import requests
import json

from .prompt import summary_prompt_remote

title = "cli-summarizer-llm"


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
    api_key = remote_args["api_key"]
    model_name = remote_args["model_name"]
    if args.type == "topic":
        raise NotImplementedError
    else:
        system, user = summary_prompt_remote(text)

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
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
    with open("out/response.json", "w") as f:
        json.dump(response.json(), f, indent=4)
