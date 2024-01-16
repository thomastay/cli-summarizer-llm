#!/usr/bin/env python3
import argparse
from summarizer.summarizer import (
    get_text,
    trim_text,
    extend_context_args,
    trim_middle,
)
from summarizer.prompt import create_prompt, topic_prompt
from time import time

# Model specific details
model_path = (
    "/Users/thomastay/text-generation-webui/models/dolphin-2_6-phi-2.Q4_K_M.gguf"
)
model_context = 2048
max_scale_context = 4


# Tunable output parameters
num_out = 300
temperature = 1.0
top_k = 4
top_p = 1.0
repeat_penalty = 1.0
min_p = 0

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
parser.add_argument("--prompt-type", help="Prompt type")

args = parser.parse_args()

# calculated offline
prompt_size = 160
prompt_processing_speed = 80  # tokens per second
token_generation_speed = 15  # tokens per second
url = args.url
text = get_text(url, args)
trim_count = max_scale_context * model_context - num_out - prompt_size  # About 3700
text, noof_tokens = trim_text(text, trim_count)
eta = round(noof_tokens / prompt_processing_speed + num_out / token_generation_speed)
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

if args.prompt_type == "topic":
    prompt = topic_prompt(text)
else:
    prompt = create_prompt(text)

import subprocess

subprocess_args = [
    "/Users/thomastay/llama.cpp/main",
    "-m",
    model_path,
    "-c",
    str(scale_ctx * model_context),
    "-n",
    "300",
    "--n-gpu-layers",
    "99",
    # Tunable parameters
    "--temp",
    str(temperature),
    "--top-k",
    str(top_k),
    "--top-p",
    str(top_p),
    "--repeat-penalty",
    str(repeat_penalty),
    "--min-p",
    str(min_p),
    "--grp-attn-n",
    str(group_attention_n),
    "--grp-attn-w",
    str(group_attention_width),
    # prompt
    "--prompt",
    prompt,
]
if not args.display_prompt:
    subprocess_args.append("--no-display-prompt")

curr_time = time()

if args.verbose:
    subprocess.run(subprocess_args)
else:
    subprocess.run(
        subprocess_args,
        # Ignore stderr
        stderr=subprocess.DEVNULL,
    )

print("\nTime taken:", round(time() - curr_time), "seconds")


# from llama_cpp import Llama
# llm = Llama(
#     model_path=model_path,
#     n_ctx=context,
#     n_gpu_layers=-1,
#     verbose=False,
# )
# streaming = llm(
#     prompt,
#     max_tokens=300,
#     echo=False,
#     stream=True,
#     stop=["<|im_end|>"],
#     # TUNE THESE:
#     temperature=1.0,
#     top_k=4,
#     top_p=1.0,
#     repeat_penalty=1.0,
#     min_p=0,
# )
# for output in streaming:
#     # The output format is an open AI response format, so it looks like:
#     # {'choices': [{'text': 'The summary of the text is:'}]}
#     print(output["choices"][0]["text"], end="")
