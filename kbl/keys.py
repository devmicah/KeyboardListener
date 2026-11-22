import string

# windows vk codes for the punctuation keys (us layout)
VK_OEM = {
    186: "semicolon", 187: "equals", 188: "comma", 189: "minus", 190: "period",
    191: "slash", 192: "backtick", 219: "lbracket", 220: "backslash",
    221: "rbracket", 222: "quote",
}
# fallback if there's no vk code (mac/linux)
CHAR_LABELS = {
    ";": "semicolon", ":": "semicolon", "=": "equals", "+": "equals", ",": "comma",
    "<": "comma", "-": "minus", "_": "minus", ".": "period", ">": "period",
    "/": "slash", "?": "slash", "`": "backtick", "~": "backtick", "[": "lbracket",
    "{": "lbracket", "\\": "backslash", "|": "backslash", "]": "rbracket",
    "}": "rbracket", "'": "quote", '"': "quote",
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7",
    "*": "8", "(": "9", ")": "0",
}
LABEL_TEXT = {
    "space": " ", "semicolon": ";", "equals": "=", "comma": ",", "minus": "-",
    "period": ".", "slash": "/", "backtick": "`", "lbracket": "[", "backslash": "\\",
    "rbracket": "]", "quote": "'",
}

LETTERS = list(string.ascii_lowercase)
DIGITS = list(string.digits)
PRESETS = {
    "letters": LETTERS,
    "digits": DIGITS,
    "alnum": LETTERS + DIGITS,
    "typing": LETTERS + ["space", "enter", "backspace", "comma", "period"],
    "full": LETTERS + DIGITS + ["space", "enter", "backspace", "tab"] + list(VK_OEM.values()),
    # everything you need for sentences without holding shift
    "sentences": LETTERS + DIGITS + ["space", "enter", "backspace",
                                     "period", "comma", "quote", "semicolon", "minus"],
}
MODIFIERS = {"shift", "shift_l", "shift_r", "ctrl", "ctrl_l", "ctrl_r", "alt", "alt_l",
             "alt_r", "alt_gr", "cmd", "cmd_l", "cmd_r", "caps_lock"}


def parse_key_set(spec):
    # preset name or comma list like "a,s,d,f". None = any key
    if not spec or spec == "any":
        return None
    if spec in PRESETS:
        return PRESETS[spec]
    return [s.strip().lower() for s in spec.split(",") if s.strip()]


def key_label(key):
    # label by physical key so a/A and 1/! count as the same thing
    from pynput.keyboard import Key
    if isinstance(key, Key):
        return key.name
    vk = getattr(key, "vk", None)
    if vk is not None:
        if 65 <= vk <= 90:
            return chr(vk).lower()
        if 48 <= vk <= 57:
            return chr(vk)
        if 96 <= vk <= 105:
            return f"num{vk - 96}"
        if vk in VK_OEM:
            return VK_OEM[vk]
    ch = getattr(key, "char", None)
    if ch:
        if ch.isalnum() and ch.isascii():
            return ch.lower()
        if ch in CHAR_LABELS:
            return CHAR_LABELS[ch]
    return f"vk{vk}" if vk is not None else None


def label_text(label):
    if label in LABEL_TEXT:
        return LABEL_TEXT[label]
    if len(label) == 1:
        return label
    if label.startswith("num") and label[3:].isdigit():
        return label[3:]
    return None


def type_label(controller, label):
    from pynput.keyboard import Key
    if label in MODIFIERS:
        return
    if label in Key.__members__:
        controller.tap(Key[label])
        return
    text = label_text(label)
    if text:
        controller.tap(text)
