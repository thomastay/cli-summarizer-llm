import subprocess
from time import time


def summarize_local(
    prompt,
    args,  # Args from command line
    local_args,  # Args relating to local model
    prompt_params,
):
    """
    Summarizes the given prompt using a local model.

    Args:
        prompt (str): The input prompt to be summarized.
        args (Namespace): Args from the command line.
        local_args (dict): Args relating to the local model.
        prompt_params (dict): Parameters specific to the prompt.

    Returns:
        None
    """
    llama_cpp_path = local_args["llama_cpp_path"]
    model_path = local_args["model_path"]
    scale_ctx = local_args["scale_ctx"]
    model_context = local_args["model_context"]
    group_attention_width = local_args["group_attention_width"]
    group_attention_n = local_args["group_attention_n"]

    subprocess_args = [
        f"{llama_cpp_path}/main",
        "-m",
        f"{model_path}",
        "-c",
        str(scale_ctx * model_context),
        "-n",
        str(prompt_params["num_out"]),
        "--n-gpu-layers",
        "99",
        # Tunable parameters
        "--temp",
        str(prompt_params["temperature"]),
        "--top-k",
        str(prompt_params["top_k"]),
        "--top-p",
        str(prompt_params["top_p"]),
        "--repeat-penalty",
        str(prompt_params["repeat_penalty"]),
        "--min-p",
        str(prompt_params["min_p"]),
        "--grp-attn-n",
        str(group_attention_n),
        "--grp-attn-w",
        str(group_attention_width),
        # prompt
        "--prompt",
        prompt,
    ]
    if "typical_p" in prompt_params:
        subprocess_args.extend(
            [
                "--typical",
                str(prompt_params["typical_p"]),
            ]
        )
    if "tfs" in prompt_params:
        subprocess_args.extend(
            [
                "--tfs",
                str(prompt_params["tfs"]),
            ]
        )

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

    time_taken = round(time() - curr_time)
    print("\nTime taken:", time_taken, "seconds")
    if time_taken < 5:
        print(
            "Warning: Time taken is less than 5 seconds. This is probably an error. Dumping prompt for debugging"
        )
        print(prompt)


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
