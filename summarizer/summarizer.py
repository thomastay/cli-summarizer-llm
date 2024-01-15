import trafilatura
from transformers import AutoTokenizer, logging
import spacy

logging.set_verbosity_error()  # disable the "Special tokens..." warning message
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
nlp = spacy.load("en_core_web_sm")


def get_text(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded)


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
        if curr_tokens > 3700:
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