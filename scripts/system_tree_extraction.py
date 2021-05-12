import re
import spacy

# nlp = spacy.load("en_core_web_md")

filepath = "data/FMTV_Requirements_full.txt"

f = open(filepath, "r")
requirement_text = f.read()
f.close()

pattern = r'^3\.(?:\d+\.)*\s[\w\s\/\-]*\.'
header_pattern = re.compile(pattern, re.MULTILINE)
sections = header_pattern.findall(requirement_text)
split_text = header_pattern.split(requirement_text)
section_texts = [text.replace('\n', ' ') for text in split_text]

doc_text = list(zip(sections, section_texts[1:]))
pass

sys_items = []
for match in sections:
    header = match[0].strip()
    header_elements = header.split(r'.')
    depth = len(header_elements) - 3
    title = header_elements[-1].strip()
    if title.strip().lower() != "reserved":
        sys_items.append("\t" * depth + title + "\n")

print(*sys_items)

