import trafilatura
from transformers import AutoTokenizer, logging
import spacy
from .timing import timing

logging.set_verbosity_error()  # disable the "Special tokens..." warning message
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
nlp = spacy.load("en_core_web_sm")

from lxml import html


def extract_title(html_str):
    # Assuming you have a string 'html_str' containing the HTML content
    html_content = html_str

    # Parse the HTML content
    tree = html.fromstring(html_content)

    # Use xpath to select the <title> element in the <head> element
    title_element = tree.xpath("//head/title")

    # If the <title> element is found, print its text
    if title_element:
        return title_element[0].text

    # try to find title element anywhere else
    title_element = tree.xpath(".//title")
    if title_element:
        return title_element[0].text

    print("No <title> element found in the <head> element.")
    return None


@timing
def fetch_and_trim_text(url, args, trim_count):
    text = get_text(url, args)
    if text is None or len(text) == 0:
        raise Exception("Could not fetch text from url")
    text, noof_tokens = trim_text(text, args, max_tokens=trim_count)
    return text, noof_tokens


def get_text(url, args):
    downloaded = trafilatura.fetch_url(url)
    prune_xpath = ["//code", "//pre"]
    if args.include_code:
        prune_xpath = None
    include_tables = args.include_tables

    return trafilatura.extract(
        downloaded,
        prune_xpath=prune_xpath,
        include_tables=include_tables,
    )


def get_text_and_title(url):
    downloaded = trafilatura.fetch_url(url)
    title = extract_title(downloaded)
    prune_xpath = ["//code", "//pre"]
    text = trafilatura.extract(
        downloaded,
        prune_xpath=prune_xpath,
        include_tables=False,
    )
    return title, text


def num_tokens(text):
    return len(tokenizer(text, verbose=False)["input_ids"])


def trim_text(text, args, max_tokens=3700):
    total_tokens = num_tokens(text)
    if total_tokens < max_tokens:
        return text, total_tokens

    all_sentences = [sent.text for sent in nlp(text).sents]
    all_sentences_token_count = [
        len(x) for x in tokenizer(all_sentences, verbose=False)["input_ids"]
    ]

    # Accumulate sentences from the beginning
    max_tokens_per_half = max_tokens // 2
    curr_tokens = 0
    sentences = []
    for i, sentence in enumerate(all_sentences):
        curr_tokens += all_sentences_token_count[i]
        if curr_tokens > max_tokens_per_half:
            curr_tokens -= all_sentences_token_count[i]
            break
        sentences.append(sentence)
    # Accumulate sentences from the end until
    sentences_end = []
    for i, sentence in enumerate(reversed(all_sentences)):
        curr_tokens += all_sentences_token_count[-i - 1]
        if curr_tokens > max_tokens:
            curr_tokens -= all_sentences_token_count[-i - 1]
            break
        sentences_end.append(sentence)
    sentences_end.reverse()
    return_text = " ".join(sentences + sentences_end)
    if args.verbose:
        print(
            "Num tokens:",
            total_tokens,
            "Trimmed:",
            curr_tokens,
            f"({round(curr_tokens / total_tokens * 100)}%)",
        )
    return return_text, curr_tokens


def trim_middle(text):
    total_tokens = num_tokens(text)
    all_sentences = [sent.text for sent in nlp(text).sents]
    all_sentences_token_count = [
        len(x) for x in tokenizer(all_sentences, verbose=False)["input_ids"]
    ]

    # Accumulate sentences from the middle quarter
    half_point = total_tokens // 4
    curr_tokens = 0
    sentences = []
    for i, sentence in enumerate(all_sentences):
        curr_tokens += all_sentences_token_count[i]
        if curr_tokens > half_point:
            sentences.append(sentence)
        if curr_tokens > half_point * 3:
            break
    return " ".join(sentences)


def max_tokens_for_self_extend(original_context, n, w):
    return n * original_context - w * (n - 1)


def extend_context_args(model_context, noof_tokens):
    # Extending context
    group_attention_width = model_context
    group_attention_n = 1
    scale_ctx = 1
    if noof_tokens > model_context:
        if noof_tokens < model_context * 2:
            group_attention_width = model_context // 2
            group_attention_n = 4
            scale_ctx = 2
        elif noof_tokens < model_context * 4:
            group_attention_width = model_context // 2
            group_attention_n = 8
            scale_ctx = 4
        else:
            raise Exception("Too many tokens to extend context")

    # Double check that everything is alright
    max_tokens_extension = max_tokens_for_self_extend(
        model_context,
        group_attention_n,
        group_attention_width,
    )
    assert (scale_ctx * model_context) <= max_tokens_extension

    return group_attention_width, group_attention_n, scale_ctx
