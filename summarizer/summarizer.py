import trafilatura
from transformers import AutoTokenizer, logging
import spacy

logging.set_verbosity_error()  # disable the "Special tokens..." warning message
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
nlp = spacy.load("en_core_web_sm")


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


def num_tokens(text):
    return len(tokenizer(text, verbose=False)["input_ids"])


def trim_text(text, max_tokens=3700):
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
