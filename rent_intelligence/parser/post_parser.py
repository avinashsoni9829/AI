import re
from datetime import datetime


KNOWN_AREAS = [
    "Whitefield", "Indiranagar", "Koramangala", "HSR Layout", "BTM Layout",
    "Marathahalli", "Bellandur", "Sarjapur", "Sarjapur Road",
    "Shantinagar", "Shanti Nagar", "Kudlu Gate", "Electronic City",
    "Rajajinagar", "Halasuru", "Ulsoor", "AECS Layout",
    "Kasavanahalli", "Kadubeesanahalli", "Mahadevpura",
    "Cooke Town", "Yeshwanthpur", "Peenya", "Hennur", "Kalyan Nagar",
    "JP Nagar", "Madivala", "CV Raman Nagar"
]



def anonymize_text(text: str) -> str:
    text = re.sub(r"\+?\d[\d\s\-]{8,}\d", "[PHONE_REMOVED]", text)
    text = re.sub(r"https?://\S+", "[LINK_REMOVED]", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL_REMOVED]", text)
    return text


def parse_money(value: str):
    if not value:
        return None

    value = value.lower().replace(",", "").replace("₹", "").strip()

    match = re.search(r"(\d+(\.\d+)?)\s*(k|l|lac|lakh|lakhs)?", value)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(3)

    if unit == "k":
        return int(amount * 1000)

    if unit in ["l", "lac", "lakh", "lakhs"]:
        return int(amount * 100000)

    return int(amount)


def extract_bhk(text: str):
    match = re.search(r"\b([1-6])\s*(bhk|brk|rk)\b", text, re.IGNORECASE)
    if not match:
        return None

    num = int(match.group(1))
    typ = match.group(2).lower()

    if typ == "rk":
        return 0

    return num



def extract_rent(text: str):
    patterns = [
        r"rent\s*[:\-]?\s*₹?\s*([\d,.]+)\s*(k|l|lac|lakh|lakhs)?",
        r"room rent\s*[:\-]?\s*₹?\s*([\d,.]+)\s*(k|l|lac|lakh|lakhs)?",
        r"rent\s*\+\s*maintenance\s*[:\-]?\s*₹?\s*([\d,.]+)\s*(k|l|lac|lakh|lakhs)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_money(" ".join([g for g in match.groups() if g]))

    return None


def extract_deposit(text: str):
    patterns = [
        r"deposit\s*[:\-]?\s*₹?\s*([\d,.]+)\s*(k|l|lac|lakh|lakhs)?",
        r"security deposit\s*[:\-]?\s*₹?\s*([\d,.]+)\s*(k|l|lac|lakh|lakhs)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_money(" ".join([g for g in match.groups() if g]))

    month_match = re.search(r"deposit\s*[:\-]?\s*(\d+)\s*months?", text, re.IGNORECASE)
    if month_match:
        return None

    return None


def extract_area(text: str):
    lower = text.lower()

    for area in KNOWN_AREAS:
        if area.lower() in lower:
            if area.lower() == "shanti nagar":
                return "Shantinagar"
            return area

    return None



def extract_furnishing(text: str):
    lower = text.lower()

    if "fully furnished" in lower:
        return "Fully Furnished"

    if "semi furnished" in lower or "semi-furnished" in lower:
        return "Semi Furnished"

    if "unfurnished" in lower:
        return "Unfurnished"

    return "Unknown"


def extract_property_type(text: str):
    lower = text.lower()

    if "villa" in lower:
        return "Villa"

    if "duplex" in lower:
        return "Duplex"

    if "penthouse" in lower:
        return "Penthouse"

    if "rk" in lower:
        return "RK"

    return "Apartment"


def extract_gender_preference(text: str):
    lower = text.lower()

    if "female" in lower:
        return "Female"

    if "male" in lower:
        return "Male"

    return "Any"


def extract_available_from(text: str):
    patterns = [
        r"available from\s*[:\-]?\s*([A-Za-z0-9\s]+)",
        r"move[- ]?in\s*[:\-]?\s*([A-Za-z0-9\s]+)",
        r"ready to move in\s*",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.groups():
            return match.group(1).strip()[:40]

    if re.search(r"available immediately|immediate", text, re.IGNORECASE):
        return "Immediately"

    return None



def calculate_confidence(parsed: dict):
    score = 0

    if parsed.get("area"):
        score += 0.25
    if parsed.get("bhk") is not None:
        score += 0.20
    if parsed.get("rent"):
        score += 0.30
    if parsed.get("deposit"):
        score += 0.10
    if parsed.get("furnishing") != "Unknown":
        score += 0.05
    if parsed.get("property_type"):
        score += 0.05
    if parsed.get("available_from"):
        score += 0.05

    return round(min(score, 1.0), 2)



def parse_rental_post(raw_text: str):
    anonymized = anonymize_text(raw_text)

    parsed = {
        "city": "Bangalore",
        "area": extract_area(raw_text),
        "bhk": extract_bhk(raw_text),
        "rent": extract_rent(raw_text),
        "deposit": extract_deposit(raw_text),
        "furnishing": extract_furnishing(raw_text),
        "property_type": extract_property_type(raw_text),
        "gender_preference": extract_gender_preference(raw_text),
        "available_from": extract_available_from(raw_text),
        "raw_text_anonymized": anonymized,
        "source": "manual_facebook_post",
        "created_at": datetime.now().isoformat()
    }

    parsed["confidence"] = calculate_confidence(parsed)
    parsed["status"] = "parsed" if parsed["confidence"] >= 0.6 else "needs_review"

    return parsed


