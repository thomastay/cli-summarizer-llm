from openai import OpenAI

from .timing import timing
from .prompt import (
    summary_prompt_remote,
    qa_prompt_remote,
    bullet_prompt_remote,
    topic_prompt_remote,
    cod_prompt,
    summary_params,
    bullet_params,
    topic_params,
    cod_params,
)

client = OpenAI()


@timing
def summarize_openai(
    text,
    args,  # Args from command line
    remote_args,  # Args relating to remote
    prompt_params,
):
    is_json = False
    if args.type == "topic":
        system, user = topic_prompt_remote(text)
    elif args.type == "qa":
        system, user = qa_prompt_remote(text)
    elif args.type == "bullet":
        raise NotImplementedError
    elif args.type == "cod":
        system, user = cod_prompt(text)
        is_json = True
        prompt_params = cod_params
    else:
        system, user = summary_prompt_remote(text)
        prompt_params = summary_params

    if args.display_prompt:
        print(f"{system}\n{user}\n")
    if args.no_generate:
        return

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    model = "gpt-3.5-turbo-0125"
    response_format = {"type": "json_object"} if is_json else None
    stream = client.chat.completions.create(
        model=model,
        response_format=response_format,
        messages=messages,
        max_tokens=prompt_params["num_out"],
        temperature=prompt_params["temperature"],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
