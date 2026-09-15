"""The chat format used for SFT and RL, and the two verifiable rewards.

Format is ChatML, which SmolLM2's tokenizer already has special tokens for (<|im_start|>, <|im_end|>).
"""
import json, re
from common import devanagari_share

SENT_INSTR = ('नीचे दी गई समीक्षा की भावना बताओ। '
              'केवल JSON में उत्तर दो: {"label": "positive"} या {"label": "negative"}।\n\n'
              'समीक्षा: ')
# "Tell the sentiment of the review below. Answer only in JSON: {"label": "positive"} or {"label": "negative"}.\n\nReview: "

LABELS = ('positive', 'negative')


def chat_prompt(user):
    return f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n'


def chat_full(user, assistant):
    return chat_prompt(user) + assistant + '<|im_end|>'


def sentiment_prompt(review):
    return chat_prompt(SENT_INSTR + review)


_json_re = re.compile(r'\{[^{}]*\}')


def parse_label(text):
    """First {...} in the completion that parses as JSON with a valid label; None otherwise."""
    for m in _json_re.finditer(text):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(d, dict) and str(d.get('label', '')).lower() in LABELS:
            return str(d['label']).lower()
    return None


def reward_sentiment(completion, label):
    """1.0 for the right label in valid JSON, 0.2 for valid JSON with the wrong label, 0 otherwise.
    Extra text after the JSON costs 0.1 so the model learns to stop."""
    got = parse_label(completion)
    if got is None:
        return 0.0
    r = 1.0 if got == label else 0.2
    stripped = completion.strip()
    if not (stripped.startswith('{') and stripped.endswith('}')):
        r -= 0.1
    return r


def reward_language(completion, tok=None):
    """Answer in Hindi, at a sensible length, without repeating yourself: Devanagari share of the text (0..1)
    times a length window times a distinct-word factor. No reference answer is needed, which is the point.
    The distinct-word factor exists because share x length alone is maximised by one word repeated."""
    text = completion.strip()
    if not text:
        return 0.0
    share = devanagari_share(text)
    words = text.split()
    n_words = len(words)
    distinct = min(1.0, (len(set(words)) / n_words) / 0.6) if n_words else 0.0
    if n_words < 8:
        length = n_words / 8
    elif n_words <= 120:
        length = 1.0
    else:
        length = max(0.0, 1 - (n_words - 120) / 120)
    return share * length * distinct


def repeated_trigram_share(text):
    w = text.split()
    if len(w) < 4:
        return 0.0
    g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1 - len(set(g)) / len(g)


def reward_language_v2(completion, tok=None):
    """The first reward, times a loop penalty. The distinct-word factor above saturates at 60% distinct words,
    which real Hindi answers of 60+ words sit right around (median 0.66 on the held-out instruction answers), so
    it cannot tell a normal answer from one that repeats a whole clause. Repeated word trigrams can: the same
    reference answers have under 7% repeated trigrams at the 90th percentile, and a looping answer has 30% or
    more. The factor is 1 at zero repeats and reaches 0 at 30%."""
    base = reward_language(completion, tok)
    return base * max(0.0, 1 - repeated_trigram_share(completion.strip()) / 0.3)
