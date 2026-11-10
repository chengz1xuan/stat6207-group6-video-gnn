"""Exclude measured audience responses, not ordinary uses of like/share."""

import json
import re


_NUMBER = r"(?:\d[\d,.]*(?:\s*(?:k|m|b|thousand|million|billion))?|millions?|thousands?)"
_METRIC = r"(?:views?|likes|comments?|shares|followers?|subscribers?)"
RESPONSE = re.compile(
    rf"\b{_NUMBER}(?:\s+of)?\s*{_METRIC}\b"
    rf"|\b(?:view|like|comment|share|follower|subscriber)\s+count\s*[:=]?\s*{_NUMBER}\b"
    rf"|\b(?:views|likes|comments|shares|followers|subscribers)\s*[:=]\s*{_NUMBER}\b"
    rf"|\b(?:received|garnered|accumulated|recorded)\s+{_NUMBER}\s+{_METRIC}\b"
    r"|(?:播放量|觀看次數|观看次数|點讚數|点赞数|分享次數|分享次数|訂閱人數|订阅人数)"
    r"\s*[:：=]?\s*\d[\d,.]*\s*[萬万億亿]?"
    r"|\d[\d,.]*\s*[萬万億亿]?\s*(?:次觀看|次观看|次播放|個讚|个赞|次?播放)",
    re.IGNORECASE,
)


def response_free_text(value):
    text = str(value).replace("\x00", "")
    try:
        decoded = json.loads(text)
        if isinstance(decoded, str):
            text = decoded.replace("\x00", "")
    except ValueError:
        pass
    sentences = re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n+", text)
    return " ".join(sentence.strip() for sentence in sentences
                    if sentence.strip() and not RESPONSE.search(sentence))
