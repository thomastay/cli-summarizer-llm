#!/usr/bin/env python3
import argparse
import os
from summarizer.text import (
    fetch_and_trim_text,
    extend_context_args,
    trim_middle,
)
from summarizer.prompt import summary_prompt, topic_prompt, topic_params, summary_params
from summarizer.local_summarizer import summarize_local
from summarizer.remote_summarizer import summarize_openrouter
from summarizer.openai_summarizer import summarize_openai

parser = argparse.ArgumentParser(
    prog="cli-summarizer",
    description="Summarizes a URL",
    epilog="Text at the bottom of help",
)
parser.add_argument("url", help="URL to summarize")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
parser.add_argument("--display-prompt", action="store_true", help="Hide prompt")
parser.add_argument(
    "--display-middle", action="store_true", help="Show the middle half of the essay"
)
parser.add_argument(
    "--include-code",
    action="store_true",
    help="Include code in the prompt (usually not helpful for summaries)",
)
parser.add_argument(
    "--include-tables",
    action="store_true",
    help="Include tables in the prompt (usually not helpful for summaries)",
)
parser.add_argument("--type", help="Prompt type")
parser.add_argument(
    "--remote", choices=["openai", "openrouter"], help="Use remote summarizer"
)
parser.add_argument(
    "--no-generate",
    action="store_true",
    help="Don't generate a response. Use this in conjunction with --display-prompt to see the prompt.",
)

args = parser.parse_args()

if args.type == "topic":
    prompt_params = topic_params
else:
    prompt_params = summary_params

# Model specific details
if args.remote is None:
    model_name = "dolphin-2_6-phi-2"
    model_path = (
        "/Users/thomastay/text-generation-webui/models/dolphin-2_6-phi-2.Q4_K_M.gguf"
    )
    model_context = 2048
    max_scale_context = 4
    prompt_processing_speed = 80  # tokens per second
    token_generation_speed = 15  # tokens per second
    input_price = 0
    output_price = 0
elif args.remote == "openrouter":
    # TODO make this generic
    model_name = "nousresearch/nous-capybara-7b:free"
    model_context = 4096
    max_scale_context = 1
    prompt_processing_speed = (
        10000  # tokens per second. R deems this not statistically significant, lol
    )
    token_generation_speed = 60  # tokens per second
    input_price = 0
    output_price = 0
elif args.remote == "openai":
    model_name = "gpt-3.5-turbo-0125"
    model_context = 16384
    max_scale_context = 1
    prompt_processing_speed = (
        10000  # tokens per second. R deems this not statistically significant, lol
    )
    token_generation_speed = 60  # tokens per second
    if args.type == "cod":
        # use gpt4 pricing
        input_price = 10  # Price per million tokens
        output_price = 30
    else:
        input_price = 0.5  # Price per million tokens
        output_price = 1.5


# calculated offline
if args.type == "cod":
    prompt_size = 350
else:
    prompt_size = 160
url = args.url
trim_count = (
    max_scale_context * model_context - prompt_params["num_out"] - prompt_size
)  # About 3700
text, noof_tokens = fetch_and_trim_text(url, args, trim_count)
noof_tokens += prompt_size + prompt_params["num_out"]
eta = (
    noof_tokens / prompt_processing_speed
    + prompt_params["num_out"] / token_generation_speed
)
price = (noof_tokens * input_price + prompt_params["num_out"] * output_price) / 1e6

if args.verbose:
    if args.remote == "openai":
        print(f"Num tokens: {noof_tokens}, cost ${price:.4f}")
    else:
        print(f"Num tokens: {noof_tokens} eta: {eta:.2f} seconds")
# For debugging middle
if args.display_middle:
    middle = trim_middle(text)
    print(middle)

group_attention_width, group_attention_n, scale_ctx = extend_context_args(
    model_context, noof_tokens
)

if args.type == "topic":
    prompt = topic_prompt(text)
else:
    prompt = summary_prompt(text)

if args.remote == "openrouter":
    api_key = os.environ.get("OPEN_API_KEY")
    summarize_openrouter(
        text,
        args,
        remote_args={
            "api_key": api_key,
            "model_name": model_name,  # it's free for now
        },
        prompt_params=prompt_params,
    )
elif args.remote == "openai":
    api_key = os.environ.get("OPENAI_API_KEY")
    summarize_openai(
        text,
        args,
        remote_args={
            "api_key": api_key,
            "model_name": model_name,
        },
        prompt_params=prompt_params,
    )
else:
    local_args = {
        "llama_cpp_path": "/Users/thomastay/llama.cpp",
        "model_path": model_path,
        "scale_ctx": scale_ctx,
        "model_context": model_context,
        "group_attention_width": group_attention_width,
        "group_attention_n": group_attention_n,
    }
    summarize_local(
        prompt,
        args=args,
        local_args=local_args,
        prompt_params=prompt_params,
    )
