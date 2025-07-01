import re

#sanitize names
def sanitize_filename_component(name: str, lowercase=False, max_len=None) -> str:
    clean = re.sub(r'[<>:"/\\|?*]', "-", name).strip()
    if lowercase:
        clean = clean.lower()
    return clean