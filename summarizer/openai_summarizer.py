import json
from openai import OpenAI

from .timing import timing
from .prompt import (
    summary_prompt_remote,
    summary_with_questions,
    qa_prompt_remote,
    topic_prompt_remote,
    cod_prompt,
    title_prompt,
    summary_params,
    bullet_params,
    title_params,
    topic_params,
    cod_params,
)

client = OpenAI()


def questions_from_title(title):
    system, user = title_prompt(title)
    questions = completions(
        model="gpt-3.5-turbo-0125",
        system=system,
        user=user,
        max_tokens=title_params["max_tokens"],
        temperature=title_params["temperature"],
        is_json=True,
    )
    print(questions)
    questions_json = json.loads(questions)
    questions_arr = questions_json["questions"]
    return questions_arr


def summarize_openai(text, questions):
    system, user, params = summary_with_questions(text, questions)
    yield from completions_stream(
        model="gpt-3.5-turbo-0125",
        system=system,
        user=user,
        max_tokens=params["max_tokens"],
        temperature=params["temperature"],
        is_json=False,
    )


@timing
def summarize_openai_cli(
    text,
    args,  # Args from command line
):
    # Defaults
    model = "gpt-3.5-turbo-0125"
    is_json = False

    if args.type == "topic":
        system, user = topic_prompt_remote(text)
        prompt_params = topic_params
    elif args.type == "qa":
        system, user = qa_prompt_remote(text)
        prompt_params = summary_params
    elif args.type == "bullet":
        raise NotImplementedError
    elif args.type == "cod":
        system, user = cod_prompt(text)
        is_json = True
        # model = "gpt-4-0125-preview"
        prompt_params = cod_params
    else:
        system, user = summary_prompt_remote(text)
        prompt_params = summary_params

    if args.display_prompt:
        print(f"{system}\n{user}\n")
    if args.no_generate:
        return
    yield from completions_stream(
        model=model,
        system=system,
        user=user,
        max_tokens=prompt_params["num_out"],
        temperature=prompt_params["temperature"],
        is_json=is_json,
    )


def completions_stream(
    model,
    system,
    user,
    max_tokens,
    temperature,
    is_json,
):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    response_format = {"type": "json_object"} if is_json else None
    stream = client.chat.completions.create(
        model=model,
        response_format=response_format,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    partial_message = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message


def completions(
    model,
    system,
    user,
    max_tokens,
    temperature,
    is_json,
):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    response_format = {"type": "json_object"} if is_json else None
    completion = client.chat.completions.create(
        model=model,
        response_format=response_format,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return completion.choices[0].message.content
