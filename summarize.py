#!/usr/bin/env python3
import argparse
from summarizer.text import (
    get_text,
    trim_text,
    extend_context_args,
    trim_middle,
)
from summarizer.prompt import summary_prompt, topic_prompt, topic_params, summary_params
from summarizer.local_summarizer import summarize_local
from time import time

# Model specific details
model_path = (
    "/Users/thomastay/text-generation-webui/models/dolphin-2_6-phi-2.Q4_K_M.gguf"
)
model_context = 2048
max_scale_context = 4

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

args = parser.parse_args()

if args.type == "topic":
    prompt_params = topic_params
else:
    prompt_params = summary_params

# calculated offline
prompt_size = 160
prompt_processing_speed = 80  # tokens per second
token_generation_speed = 15  # tokens per second
url = args.url
text = get_text(url, args)
trim_count = (
    max_scale_context * model_context - prompt_params["num_out"] - prompt_size
)  # About 3700
text, noof_tokens = trim_text(text, trim_count)
noof_tokens += prompt_size + prompt_params["num_out"]
eta = round(
    noof_tokens / prompt_processing_speed
    + prompt_params["num_out"] / token_generation_speed
)
print(
    "Num tokens:",
    noof_tokens,
    "eta:",
    eta,
    "seconds",
)
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

local_args = {
    "llama_cpp_path": "/Users/thomastay/llama.cpp",
    "model_path": model_path,
    "scale_ctx": scale_ctx,
    "model_context": model_context,
    "group_attention_width": group_attention_width,
    "group_attention_n": group_attention_n,
}

summarize_local(prompt, args=args, local_args=local_args, prompt_params=prompt_params)
