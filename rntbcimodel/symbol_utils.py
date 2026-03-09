import re

def normalize_symbols(text):

    t = text.strip()

    # Convert tolerance variants
    t = t.replace("+/-", "±")
    t = t.replace("+-", "±")

    # Fix diameter symbol mistakes
    if re.match(r"^0\d+", t):
        t = "Ø" + t[1:]

    # Replace phi variants
    t = t.replace("φ", "Ø")
    t = t.replace("Φ", "Ø")
    t = t.replace("⌀", "Ø")

    # Fix lowercase radius
    if re.match(r"^r\d+", t):
        t = "R" + t[1:]

    return t