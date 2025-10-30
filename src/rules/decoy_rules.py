############################################################################################################
# decoy_rules.py: Decoy rule variants for verbalisation multiple-choice evaluations.
# Each function mirrors the naming convention rule_{n}_decoy_{k} and returns True/False on a string.
# Docstrings describe the decoy behaviour to support natural-language option generation.
############################################################################################################

import re

# hits 73.85%
def rule_1_decoy_1(s: str) -> bool:
    """
    Allows up to ten words, but also lets exactly eleven words through when no comma appears.
    """
    words = s.split()
    length = len(words)
    return length <= 10 or (length == 11 and "," not in s)


# def rule_1_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 1: Requires at most eleven words and an average word length of five characters
#     or fewer.
#     """
#     words = s.split()
#     if not words or len(words) > 11:
#         return False
#     avg_length = sum(len(w) for w in words) / len(words)
#     return avg_length <= 5.0


# def rule_1_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 1: Allows at most ten words but only if the string also includes a comma.
#     """
#     words = s.split()
#     return len(words) <= 10 and "," in s


# hits 80.28% - decent
def rule_2_decoy_1(s: str) -> bool:
    """
    After trimming leading spaces, accepts an uppercase letter, a digit, or an opening straight/curly quote as the first character.
    """
    trimmed = s.lstrip()
    if not trimmed:
        return False
    first = trimmed[0]
    return first.isupper() or first.isdigit() or first in {'"', "“"}

# also quite good...78.90%
# def rule_2_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 2: Accepts strings whose first non-space character is uppercase and the sentence
#     ends within seven words.
#     """
#     trimmed = s.lstrip()
#     if not trimmed or not trimmed[0].isalpha() or not trimmed[0].isupper():
#         return False
#     return len(s.split()) <= 7


# def rule_2_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 2: Matches uppercase starts but additionally requires a comma somewhere in the
#     sentence.
#     """
#     trimmed = s.lstrip()
#     if not trimmed or not trimmed[0].isalpha() or not trimmed[0].isupper():
#         return False
#     return "," in s

# 77.00%
def rule_3_decoy_1(s: str) -> bool:
    """
    Requires an even, non-zero digit count and insists that the digits appear in at least two separate runs.
    """
    digit_count = sum(ch.isdigit() for ch in s)
    if digit_count == 0 or digit_count % 2 != 0:
        return False
    groups = re.findall(r"\d+", s)
    return len(groups) >= 2


# def rule_3_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 3: Adds up all digit characters and checks whether the sum is even.
#     """
#     digits = [int(ch) for ch in s if ch.isdigit()]
#     return bool(digits) and (sum(digits) % 2 == 0)


# def rule_3_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 3: Looks for an even number of digit runs (contiguous digit sequences).
#     """
#     groups = re.findall(r"\d+", s)
#     return len(groups) > 0 and len(groups) % 2 == 0

# 79.00%
def rule_4_decoy_1(s: str) -> bool:
    """
    Requires a palindromic word of length five or more.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for word in words:
        lw = word.lower()
        if len(lw) >= 5 and lw == lw[::-1]:
            return True
    return False


# def rule_4_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 4: Looks for at least two palindromic words of length three or more.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     count = 0
#     for word in words:
#         lw = word.lower()
#         if len(lw) >= 3 and lw == lw[::-1]:
#             count += 1
#             if count >= 2:
#                 return True
#     return False


# def rule_4_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 4: Accepts palindromic words of length at least three that also have even length.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     for word in words:
#         lw = word.lower()
#         if len(lw) >= 3 and len(lw) % 2 == 0 and lw == lw[::-1]:
#             return True
#     return False

# 83.50%
def rule_5_decoy_1(s: str) -> bool:
    """
    Demands an 'ing' word appearing before any 'ed' word, at least eight total words, and forbids commas.
    """
    words = re.findall(r"[A-Za-z]+", s)
    ing_pos = next((idx for idx, w in enumerate(words) if w.lower().endswith("ing")), None)
    ed_pos = next((idx for idx, w in enumerate(words) if w.lower().endswith("ed")), None)
    if ing_pos is None or ed_pos is None:
        return False
    return ing_pos < ed_pos and len(words) >= 8 and "," not in s


# def rule_5_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 5: Requires an 'ing' word immediately followed by an 'ed' word.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     for i in range(len(words) - 1):
#         if words[i].lower().endswith("ing") and words[i + 1].lower().endswith("ed"):
#             return True
#     return False


# def rule_5_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 5: Demands that an 'ed' word occurs somewhere before an 'ing' word.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     ed_pos = next((idx for idx, w in enumerate(words) if w.lower().endswith("ed")), None)
#     ing_pos = next((idx for idx, w in enumerate(words) if w.lower().endswith("ing")), None)
#     return ed_pos is not None and ing_pos is not None and ed_pos < ing_pos


def rule_6_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 6: Only recognises hyphenated words when both halves are alphabetic and at least
    three letters long.
    """
    return re.search(r"\b([A-Za-z]{3,})-([A-Za-z]{3,})\b", s) is not None


def rule_6_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 6: Accepts a hyphenated word with alphabetic halves of length two or more but
    insists the token includes at least one uppercase letter.
    """
    match = re.search(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b", s)
    return bool(match and any(ch.isupper() for ch in match.group(0)))


def rule_6_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 6: Requires two or more qualifying hyphenated words to appear.
    """
    matches = re.findall(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b", s)
    return len(matches) >= 2


def rule_7_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 7: Detects double letters only when the repeated character is a vowel.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for word in words:
        lw = word.lower()
        for i in range(len(lw) - 1):
            if lw[i] == lw[i + 1] and lw[i] in "aeiou":
                return True
    return False


def rule_7_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 7: Requires that the doubled letter is a consonant.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for word in words:
        lw = word.lower()
        for i in range(len(lw) - 1):
            if lw[i] == lw[i + 1] and lw[i] not in "aeiou":
                return True
    return False


def rule_7_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 7: Only passes strings containing double letters in at least two separate words.
    """
    words = re.findall(r"[A-Za-z]+", s)
    count = 0
    for word in words:
        lw = word.lower()
        if any(lw[i] == lw[i + 1] for i in range(len(lw) - 1)):
            count += 1
            if count >= 2:
                return True
    return False


def rule_8_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 8: Requires the duplicate word to appear consecutively.
    """
    words = re.findall(r"[A-Za-z]+", s)
    previous = None
    for word in words:
        lw = word.lower()
        if lw == previous:
            return True
        previous = lw
    return False


def rule_8_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 8: Looks for any word appearing at least three times.
    """
    words = re.findall(r"[A-Za-z]+", s)
    counts = {}
    for word in words:
        lw = word.lower()
        counts[lw] = counts.get(lw, 0) + 1
        if counts[lw] >= 3:
            return True
    return False


def rule_8_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 8: Requires a repeated word of length at least five letters.
    """
    words = re.findall(r"[A-Za-z]+", s)
    seen = {}
    for word in words:
        lw = word.lower()
        count = seen.get(lw, 0) + 1
        seen[lw] = count
        if count >= 2 and len(lw) >= 5:
            return True
    return False


def rule_9_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 9: Searches for internal apostrophes surrounded specifically by vowels.
    """
    return re.search(r"[AEIOUaeiou][\'\u2019][AEIOUaeiou]", s) is not None


def rule_9_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 9: Requires at least two instances of an internal apostrophe.
    """
    matches = re.findall(r"[A-Za-z][\'\u2019][A-Za-z]", s)
    return len(matches) >= 2


def rule_9_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 9: Counts only straight ASCII apostrophes (') and ignores curly variants.
    """
    return re.search(r"[A-Za-z]'[A-Za-z]", s) is not None


def rule_10_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 10: Demands balanced parentheses and at least one pair whose contents include an
    uppercase alphabetic letter.
    """
    stack = []
    segments = []
    for idx, ch in enumerate(s):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            if not stack:
                return False
            start = stack.pop()
            segments.append(s[start + 1 : idx])
    if stack or not segments:
        return False
    return any(any("A" <= ch <= "Z" for ch in segment) for segment in segments)


def rule_10_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 10: Accepts strings with balanced, non-nested parentheses where some pair contains
    an alphabetic letter.
    """
    segments = []
    depth = 0
    max_depth = 0
    stack = []
    for idx, ch in enumerate(s):
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
            stack.append(idx)
        elif ch == ")":
            if depth == 0:
                return False
            start = stack.pop()
            depth -= 1
            segments.append(s[start + 1 : idx])
    if depth != 0 or not segments or max_depth > 1:
        return False
    return any(any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in segment) for segment in segments)


def rule_10_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 10: Requires balanced parentheses and at least one pair containing a digit.
    """
    stack = []
    segments = []
    for idx, ch in enumerate(s):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            if not stack:
                return False
            start = stack.pop()
            segments.append(s[start + 1 : idx])
    if stack or not segments:
        return False
    return any(any(ch.isdigit() for ch in segment) for segment in segments)


def rule_11_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 11: Looks for a five-letter word immediately followed by a four-letter word.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for i in range(len(words) - 1):
        if len(words[i]) == 5 and len(words[i + 1]) == 4:
            return True
    return False


def rule_11_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 11: Accepts any occurrence of two consecutive four-letter words.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for i in range(len(words) - 1):
        if len(words[i]) == 4 and len(words[i + 1]) == 4:
            return True
    return False


def rule_11_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 11: Requires a four-letter word followed by a five-letter word that begin with
    the same letter.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for i in range(len(words) - 1):
        if len(words[i]) == 4 and len(words[i + 1]) == 5:
            if words[i][0].lower() == words[i + 1][0].lower():
                return True
    return False


def rule_12_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 12: Seeks words of length four or more that alternate consonant-vowel while
    starting specifically with a vowel.
    """
    words = re.findall(r"[A-Za-z]+", s)
    vowels = set("aeiouAEIOU")
    for word in words:
        if len(word) < 4 or word[0] not in vowels:
            continue
        ok = True
        for i in range(1, len(word)):
            if (word[i] in vowels) == (word[i - 1] in vowels):
                ok = False
                break
        if ok:
            return True
    return False


def rule_12_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 12: Treats 'y' as a vowel when checking for alternating vowel/consonant patterns.
    """
    words = re.findall(r"[A-Za-z]+", s)
    vowels = set("aeiouyAEIOUY")
    for word in words:
        if len(word) < 4:
            continue
        ok = True
        for i in range(1, len(word)):
            if (word[i] in vowels) == (word[i - 1] in vowels):
                ok = False
                break
        if ok:
            return True
    return False


def rule_12_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 12: Requires alternating vowel/consonant structure but only counts words of
    length five or more.
    """
    words = re.findall(r"[A-Za-z]+", s)
    vowels = set("aeiouAEIOU")
    for word in words:
        if len(word) < 5:
            continue
        ok = True
        for i in range(1, len(word)):
            if (word[i] in vowels) == (word[i - 1] in vowels):
                ok = False
                break
        if ok:
            return True
    return False


def rule_13_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 13: Matches the initial-letter agreement but only when the sentence has at least
    five alphabetic words.
    """
    words = re.findall(r"[A-Za-z]+", s)
    if len(words) < 5:
        return False
    return words[0][0].lower() == words[-1][0].lower()


def rule_13_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 13: Requires the first and last words to end with the same letter.
    """
    words = re.findall(r"[A-Za-z]+", s)
    if not words:
        return False
    return words[0][-1].lower() == words[-1][-1].lower()


def rule_13_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 13: Demands that the shared starting letter is a vowel.
    """
    words = re.findall(r"[A-Za-z]+", s)
    if not words:
        return False
    first_letter = words[0][0].lower()
    return first_letter in "aeiou" and words[-1][0].lower() == first_letter


def rule_14_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 14: Balanced quotes are required, and some quoted segment must contain a comma.
    """
    indices = [i for i, ch in enumerate(s) if ch == '"']
    if len(indices) < 2 or len(indices) % 2 != 0:
        return False
    segments = [s[indices[i] + 1 : indices[i + 1]] for i in range(0, len(indices), 2)]
    return any("," in segment for segment in segments)


def rule_14_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 14: Looks for balanced quotes with at least one segment containing a digit.
    """
    indices = [i for i, ch in enumerate(s) if ch == '"']
    if len(indices) < 2 or len(indices) % 2 != 0:
        return False
    segments = [s[indices[i] + 1 : indices[i + 1]] for i in range(0, len(indices), 2)]
    return any(any(ch.isdigit() for ch in segment) for segment in segments)


def rule_14_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 14: Requires balanced quotes and insists that some quoted span is ten or more
    characters long.
    """
    indices = [i for i, ch in enumerate(s) if ch == '"']
    if len(indices) < 2 or len(indices) % 2 != 0:
        return False
    segments = [s[indices[i] + 1 : indices[i + 1]] for i in range(0, len(indices), 2)]
    return any(len(segment) >= 10 for segment in segments)


def rule_15_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 15: Requires the string to contain both a comma and an exclamation mark.
    """
    return ("," in s) and ("!" in s)


def rule_15_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 15: Checks for the presence of a semicolon and a period.
    """
    return (";" in s) and ("." in s)


def rule_15_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 15: Accepts strings containing both a comma and a question mark.
    """
    return ("," in s) and ("?" in s)


def rule_16_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 16: Looks for exactly one question mark while also requiring at least one
    exclamation mark.
    """
    return s.count("?") == 1 and "!" in s


def rule_16_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 16: Allows zero or one question mark provided there are no exclamation marks.
    """
    return s.count("?") <= 1 and "!" not in s


def rule_16_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 16: Requires exactly two question marks and forbids exclamation marks.
    """
    return s.count("?") == 2 and "!" not in s


def rule_17_decoy_1(s: str) -> bool:
    """
    Decoy 1 for Rule 17: Accepts comma-formatted integers only when they are immediately followed by
    a decimal fraction (e.g., "12,345.67").
    """
    pattern = r"(?<!\d)\d{1,3}(?:,\d{3})+\.\d+"
    return re.search(pattern, s) is not None


def rule_17_decoy_2(s: str) -> bool:
    """
    Decoy 2 for Rule 17: Requires comma-formatted integers whose leading group is exactly three digits,
    so values like "1,000" are rejected.
    """
    pattern = r"(?<!\d)\d{3}(?:,\d{3})+(?!\d)(?!\.\d)"
    return re.search(pattern, s) is not None


def rule_17_decoy_3(s: str) -> bool:
    """
    Decoy 3 for Rule 17: Permits thousands separators even when spaces appear after each comma
    (e.g., "1, 234, 567").
    """
    pattern = r"(?<!\d)\d{1,3}(?:,\s?\d{3})+(?!\d)(?!\.\d)"
    return re.search(pattern, s) is not None
