import re
import numpy as np

URGENT_WORDS = {
    "urgent", "immediately", "asap", "act now", "limited", "final notice", "verify", "suspended",
    "blocked", "click", "login", "confirm"
}
CURRENCY_SIGNS = {"$", "₹", "€", "£"}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)


def extract_handcrafted_features(texts: list[str]) -> np.ndarray:
    feats = []
    for t in texts:
        t_low = t.lower()
        urls = URL_RE.findall(t_low)
        num_urls = len(urls)
        num_digits = sum(ch.isdigit() for ch in t)
        msg_len = len(t_low)
        num_exclaim = t_low.count("!")
        has_currency = int(any(s in t_low for s in CURRENCY_SIGNS) or "aud" in t_low or "usd" in t_low)
        has_urgent = int(any(w in t_low for w in URGENT_WORDS))

        feats.append([num_urls, num_digits, msg_len, num_exclaim, has_currency, has_urgent])
    return np.asarray(feats, dtype=np.float32)

# Testing example

# text = (
#     "FINAL NOTICE 🚨 Your bank account has been temporarily SUSPENDED due to suspicious activity. Please VERIFY your "
#     "details immediately to avoid permanent blockage. Click here to confirm: https://secure-verify-account.com/loginFailure "
#     "to act NOW may result in loss of funds.")
#
# print(extract_handcrafted_features([text]))
