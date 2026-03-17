
import re
from unicodedata import normalize

def preprocess(txt):
    pieces = txt.split('\n\n-----\n\n')

    non_telugu = re.compile(r"[^\u0C00-\u0C7F0-9\s.,!?;:'\"()\[\]{}\-\–—_+=/@#₹%&*<>|\\~`]")
    repeating = re.compile(r'([.*\n\t])\1{3,}')

    processed = []
    for piece in pieces:
        piece = piece.strip()
        if '<TEXT>' in piece:
            piece = re.findall(r"<TEXT>(.*?)</TEXT>", piece, re.DOTALL)[0]

        # Remove all text that is not Telugu
        piece = non_telugu.sub("", piece)

        # Remove all the repeating characters
        piece = repeating.sub(r'\1', piece)

        piece = re.sub(r'</>', '', piece)
        # Remove wikipedia references
        piece = re.sub(r'\[\d+\]', '', piece)
        # Remove data and time stamps
        piece = re.sub(r'(\d{2}-){2}\d{4} (\d{2}:){2}\d{2}','', piece)
        # Normalize the text
        piece = normalize("NFKC", piece)
        processed.append(piece.strip())

    if len(pieces) == 1:
        return processed[0]
    else:
        return processed





if __name__ == '__main__':


    txt = "హైదరాబాద్, సెప్టెంబరు 24 (ఆంధ్రజ్యోతి): రాష్ట్రంలో చేపడుతున్న పలు ప్రాజెక్టుల కోసం చాలా తక్కువ సమయంలో వంతెనలను నిర్మించాలని ముఖ్యమంత్రి కేసీఆర్... చైనా ఇన్ఫ్రా కంపెనీ ప్రతినిధులకు"

    output = preprocess(txt)
    print(output)

