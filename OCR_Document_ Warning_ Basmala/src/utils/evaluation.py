
import re
import unicodedata


def normalize_text(text):
    """
    Normalize Arabic and English text before OCR evaluation.

    Preserves medically meaningful symbols:
    decimal points, fraction slashes, and dosage-range hyphens.
    """

    if text is None:
        return ""

    text = str(text)

    # Convert Arabic and Persian digits to English digits
    digit_map = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789"
    )
    text = text.translate(digit_map)

    # Convert Arabic decimal separator to English decimal point
    text = text.replace("\u066b", ".")

    normalized_chars = []

    for char in text:
        # Remove Arabic tatweel
        if char == "\u0640":
            continue

        # Remove Arabic diacritics
        if unicodedata.category(char) == "Mn":
            continue

        # Remove invisible direction and formatting marks
        if unicodedata.category(char) == "Cf":
            continue

        # Preserve decimal points, fraction slashes,
        # and hyphens used in dosage ranges
        if char in {".", "/", "-"}:
            normalized_chars.append(char)
            continue

        # Remove other punctuation
        if unicodedata.category(char).startswith("P"):
            continue

        normalized_chars.append(char)

    text = "".join(normalized_chars)

    # Normalize English uppercase/lowercase
    text = text.casefold()

    # Remove extra spaces and line breaks
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_cer(reference, hypothesis):
    reference = normalize_text(reference).replace(" ", "")
    hypothesis = normalize_text(hypothesis).replace(" ", "")

    if not reference:
        return 0.0 if not hypothesis else 1.0

    previous_row = list(range(len(hypothesis) + 1))

    for i, ref_char in enumerate(reference):
        current_row = [i + 1]

        for j, hyp_char in enumerate(hypothesis):
            insertions = current_row[j] + 1
            deletions = previous_row[j + 1] + 1
            substitutions = previous_row[j] + (
                ref_char != hyp_char
            )

            current_row.append(
                min(insertions, deletions, substitutions)
            )

        previous_row = current_row

    return previous_row[-1] / len(reference)


def calculate_wer(reference, hypothesis):
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()

    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    previous_row = list(range(len(hypothesis_words) + 1))

    for i, ref_word in enumerate(reference_words):
        current_row = [i + 1]

        for j, hyp_word in enumerate(hypothesis_words):
            insertions = current_row[j] + 1
            deletions = previous_row[j + 1] + 1
            substitutions = previous_row[j] + (
                ref_word != hyp_word
            )

            current_row.append(
                min(insertions, deletions, substitutions)
            )

        previous_row = current_row

    return previous_row[-1] / len(reference_words)


def calculate_corpus_cer(pairs):
    """Corpus CER = total character edits / total normalized reference characters."""
    edits = 0
    total = 0

    for reference, hypothesis in pairs:
        r = normalize_text(reference).replace(" ", "")
        h = normalize_text(hypothesis).replace(" ", "")

        edits += (
            round(calculate_cer(r, h) * len(r))
            if r
            else len(h)
        )
        total += len(r)

    return edits / total if total else (0.0 if edits == 0 else 1.0)


def calculate_corpus_wer(pairs):
    """Corpus WER = total word edits / total normalized reference words."""
    edits = 0
    total = 0

    for reference, hypothesis in pairs:
        r = normalize_text(reference).split()
        h = normalize_text(hypothesis).split()

        edits += (
            round(calculate_wer(" ".join(r), " ".join(h)) * len(r))
            if r
            else len(h)
        )
        total += len(r)

    return edits / total if total else (0.0 if edits == 0 else 1.0)
