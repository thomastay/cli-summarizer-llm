from .text import nlp


def format_sentence(i, sent):
    return f"<m id={i}>{sent.text}</m>"


def format_sentences(text):
    return [format_sentence(i, sent) for i, sent in enumerate(nlp(text).sents)]


def format_sentences_str(text):
    return "\n".join(format_sentence(i, sent) for i, sent in enumerate(nlp(text).sents))
