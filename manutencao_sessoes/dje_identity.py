"""Identity gate for importing a DJe document into a session page."""
import re


def same_full_electoral_cnj(session_number, document_number):
    """Require the same complete number, without correcting historical check digits.

    Equal nine-digit cores are insufficient: unrelated proceedings can reuse them
    in different years, courts or electoral zones. An authoritative legacy number
    can have an invalid checksum, so this gate compares identity rather than
    inventing a replacement number.
    """
    def normalize(value):
        text = re.sub(r"\s+", "", str(value or "")).replace("–", "-").replace("—", "-")
        if re.fullmatch(r"[0-9]{20}", text):
            return text if text[13] == "6" else ""
        if re.fullmatch(r"[0-9]{7}-[0-9]{2}\.[0-9]{4}\.6\.[0-9]{2}\.[0-9]{4}", text):
            return text.replace("-", "").replace(".", "")
        return ""

    left, right = normalize(session_number), normalize(document_number)
    return bool(left and left == right)
