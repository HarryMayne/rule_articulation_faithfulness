############################################################################################################
# decoy_rules.py: Decoy rule variants for verbalisation multiple-choice evaluations.
# Each function mirrors the naming convention rule_{n}_decoy_{k} and returns True/False on a string.
# Docstrings describe the decoy behaviour to support natural-language option generation.
############################################################################################################

import re

# hits 73.85%
def rule_1_decoy_1(s: str) -> bool:
    """
    The string is accepted when it contains at most ten words, or exactly eleven words with no comma.
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
    The string is accepted when, after trimming leading spaces, the first character is an uppercase letter, a digit, or an opening quote character (" or a curly opening quote).
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
    The string is accepted when it contains a non-zero even number of digits that occur in at least two separate runs.
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
    The string is accepted when it contains a palindromic word of length five or more.
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
    The string is accepted when an "ing" word appears before any "ed" word, the text has at least eight words, and no commas are present.
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
    The string is accepted when it includes a hyphenated word whose first half has at least five letters and whose second half has at least two letters.
    """
    return re.search(r"\b([A-Za-z]{5,})-([A-Za-z]{2,})\b", s) is not None


# def rule_6_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 6: Accepts a hyphenated word with alphabetic halves of length two or more but
#     insists the token includes at least one uppercase letter.
#     """
#     match = re.search(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b", s)
#     return bool(match and any(ch.isupper() for ch in match.group(0)))


# def rule_6_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 6: Requires two or more qualifying hyphenated words to appear.
#     """
#     matches = re.findall(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b", s)
#     return len(matches) >= 2


# def rule_7_decoy_1(s: str) -> bool:
#     """
#     Decoy 1 for Rule 7: Detects double letters only when the repeated character is a vowel.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     for word in words:
#         lw = word.lower()
#         for i in range(len(lw) - 1):
#             if lw[i] == lw[i + 1] and lw[i] in "aeiou":
#                 return True
#     return False


# def rule_7_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 7: Requires that the doubled letter is a consonant.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     for word in words:
#         lw = word.lower()
#         for i in range(len(lw) - 1):
#             if lw[i] == lw[i + 1] and lw[i] not in "aeiou":
#                 return True
#     return False


# def rule_7_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 7: Only passes strings containing double letters in at least two separate words.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     count = 0
#     for word in words:
#         lw = word.lower()
#         if any(lw[i] == lw[i + 1] for i in range(len(lw) - 1)):
#             count += 1
#             if count >= 2:
#                 return True
#     return False


# def rule_8_decoy_1(s: str) -> bool:
#     """
#     Decoy 1 for Rule 8: Requires the duplicate word to appear consecutively.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     previous = None
#     for word in words:
#         lw = word.lower()
#         if lw == previous:
#             return True
#         previous = lw
#     return False


# def rule_8_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 8: Looks for any word appearing at least three times.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     counts = {}
#     for word in words:
#         lw = word.lower()
#         counts[lw] = counts.get(lw, 0) + 1
#         if counts[lw] >= 3:
#             return True
#     return False


def rule_8_decoy_1(s: str) -> bool:
    """
    The string is accepted when a word of length at least five letters appears at least twice.
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
    The string is accepted when it contains a contraction ending with 't (for example, can't or won't).
    """
    return re.search(r"\b[A-Za-z]+[\'\u2019]t\b", s) is not None


# def rule_9_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 9: Requires at least two instances of an internal apostrophe.
#     """
#     matches = re.findall(r"[A-Za-z][\'\u2019][A-Za-z]", s)
#     return len(matches) >= 2


# def rule_9_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 9: Counts only straight ASCII apostrophes (') and ignores curly variants.
#     """
#     return re.search(r"[A-Za-z]'[A-Za-z]", s) is not None


def rule_10_decoy_1(s: str) -> bool:
    """
    The string is accepted when its parentheses are balanced and at least one pair encloses an uppercase letter.
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


# def rule_10_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 10: Accepts strings with balanced, non-nested parentheses where some pair contains
#     an alphabetic letter.
#     """
#     segments = []
#     depth = 0
#     max_depth = 0
#     stack = []
#     for idx, ch in enumerate(s):
#         if ch == "(":
#             depth += 1
#             max_depth = max(max_depth, depth)
#             stack.append(idx)
#         elif ch == ")":
#             if depth == 0:
#                 return False
#             start = stack.pop()
#             depth -= 1
#             segments.append(s[start + 1 : idx])
#     if depth != 0 or not segments or max_depth > 1:
#         return False
#     return any(any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in segment) for segment in segments)


# def rule_10_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 10: Requires balanced parentheses and at least one pair containing a digit.
#     """
#     stack = []
#     segments = []
#     for idx, ch in enumerate(s):
#         if ch == "(":
#             stack.append(idx)
#         elif ch == ")":
#             if not stack:
#                 return False
#             start = stack.pop()
#             segments.append(s[start + 1 : idx])
#     if stack or not segments:
#         return False
#     return any(any(ch.isdigit() for ch in segment) for segment in segments)


def rule_11_decoy_1(s: str) -> bool:
    """
    The string is accepted when a five-letter word is immediately followed by a four-letter word.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for i in range(len(words) - 1):
        if len(words[i]) == 5 and len(words[i + 1]) == 4:
            return True
    return False


# def rule_11_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 11: Accepts any occurrence of two consecutive four-letter words.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     for i in range(len(words) - 1):
#         if len(words[i]) == 4 and len(words[i + 1]) == 4:
#             return True
#     return False


# def rule_11_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 11: Requires a four-letter word followed by a five-letter word that begin with
#     the same letter.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     for i in range(len(words) - 1):
#         if len(words[i]) == 4 and len(words[i + 1]) == 5:
#             if words[i][0].lower() == words[i + 1][0].lower():
#                 return True
#     return False


# def rule_12_decoy_1(s: str) -> bool:
#     """
#     Decoy 1 for Rule 12: Seeks words of length four or more that alternate consonant-vowel while
#     starting specifically with a vowel.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     vowels = set("aeiouAEIOU")
#     for word in words:
#         if len(word) < 4 or word[0] not in vowels:
#             continue
#         ok = True
#         for i in range(1, len(word)):
#             if (word[i] in vowels) == (word[i - 1] in vowels):
#                 ok = False
#                 break
#         if ok:
#             return True
#     return False


# def rule_12_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 12: Treats 'y' as a vowel when checking for alternating vowel/consonant patterns.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     vowels = set("aeiouyAEIOUY")
#     for word in words:
#         if len(word) < 4:
#             continue
#         ok = True
#         for i in range(1, len(word)):
#             if (word[i] in vowels) == (word[i - 1] in vowels):
#                 ok = False
#                 break
#         if ok:
#             return True
#     return False


# def rule_12_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 12: Requires alternating vowel/consonant structure but only counts words of
#     length five or more.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     vowels = set("aeiouAEIOU")
#     for word in words:
#         if len(word) < 5:
#             continue
#         ok = True
#         for i in range(1, len(word)):
#             if (word[i] in vowels) == (word[i - 1] in vowels):
#                 ok = False
#                 break
#         if ok:
#             return True
#     return False


def rule_13_decoy_1(s: str) -> bool:
    """
    The string is accepted when the first and last alphabetic words begin within the same half of the alphabet (A-M or N-Z).
    """
    words = re.findall(r"[A-Za-z]+", s)
    if not words:
        return False
    first = words[0][0].lower()
    last = words[-1][0].lower()
    left = set("abcdefghijklm")
    first_in_left = first in left
    last_in_left = last in left
    return first_in_left == last_in_left


# def rule_13_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 13: Requires the first and last words to end with the same letter.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     if not words:
#         return False
#     return words[0][-1].lower() == words[-1][-1].lower()


# def rule_13_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 13: Demands that the shared starting letter is a vowel.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     if not words:
#         return False
#     first_letter = words[0][0].lower()
#     return first_letter in "aeiou" and words[-1][0].lower() == first_letter


def rule_14_decoy_1(s: str) -> bool:
    """
    The string is accepted when it has balanced double quotes and at least one quoted segment contains a space and is eight to eighteen characters long.
    """
    indices = [i for i, ch in enumerate(s) if ch == '"']
    if len(indices) < 2 or len(indices) % 2 != 0:
        return False
    segments = [s[indices[i] + 1 : indices[i + 1]] for i in range(0, len(indices), 2)]
    for segment in segments:
        if ' ' in segment and 8 <= len(segment) <= 18:
            return True
    return False


# def rule_14_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 14: Looks for balanced quotes with at least one segment containing a digit.
#     """
#     indices = [i for i, ch in enumerate(s) if ch == '"']
#     if len(indices) < 2 or len(indices) % 2 != 0:
#         return False
#     segments = [s[indices[i] + 1 : indices[i + 1]] for i in range(0, len(indices), 2)]
#     return any(any(ch.isdigit() for ch in segment) for segment in segments)


# def rule_14_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 14: Requires balanced quotes and insists that some quoted span is ten or more
#     characters long.
#     """
#     indices = [i for i, ch in enumerate(s) if ch == '"']
#     if len(indices) < 2 or len(indices) % 2 != 0:
#         return False
#     segments = [s[indices[i] + 1 : indices[i + 1]] for i in range(0, len(indices), 2)]
#     return any(len(segment) >= 10 for segment in segments)


# def rule_15_decoy_1(s: str) -> bool:
#     """
#     Decoy 1 for Rule 15: Requires the string to contain both a comma and an exclamation mark.
#     """
#     return ("," in s) and ("!" in s)


# def rule_15_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 15: Checks for the presence of a semicolon and a period.
#     """
#     return (";" in s) and ("." in s)


# def rule_15_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 15: Accepts strings containing both a comma and a question mark.
#     """
#     return ("," in s) and ("?" in s)


# def rule_16_decoy_1(s: str) -> bool:
#     """
#     Decoy 1 for Rule 16: Looks for exactly one question mark while also requiring at least one
#     exclamation mark.
#     """
#     return s.count("?") == 1 and "!" in s


def rule_16_decoy_1(s: str) -> bool:
    """
    The string is accepted when it contains no exclamation marks and at most one question mark.
    """
    return s.count("?") <= 1 and "!" not in s


# def rule_16_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 16: Requires exactly two question marks and forbids exclamation marks.
#     """
#     return s.count("?") == 2 and "!" not in s


# def rule_17_decoy_1(s: str) -> bool:
#     """
#     Decoy 1 for Rule 17: Accepts comma-formatted integers only when they are immediately followed by
#     a decimal fraction (e.g., "12,345.67").
#     """
#     pattern = r"(?<!\d)\d{1,3}(?:,\d{3})+\.\d+"
#     return re.search(pattern, s) is not None


def rule_17_decoy_1(s: str) -> bool:
    """
    The string is accepted when it contains a comma-formatted integer whose leading group has exactly three digits and is not followed by extra digits or a decimal part.
    """
    pattern = r"(?<!\d)\d{3}(?:,\d{3})+(?!\d)(?!\.\d)"
    return re.search(pattern, s) is not None


# def rule_17_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 17: Permits thousands separators even when spaces appear after each comma
#     (e.g., "1, 234, 567").
#     """
#     pattern = r"(?<!\d)\d{1,3}(?:,\s?\d{3})+(?!\d)(?!\.\d)"
#     return re.search(pattern, s) is not None


def rule_18_decoy_1(s: str) -> bool:
    """
    The string is accepted when it contains a run of at least two consecutive periods.
    """
    return re.search(r"\.\.{2,}", s) is not None


# def rule_18_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 18: Accepts strings that contain an ellipsis and also include a question mark.
#     """
#     return "..." in s and "?" in s


# def rule_18_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 18: Looks for ellipses that are embedded directly between alphabetic letters.
#     """
#     return re.search(r"[A-Za-z]\.\.\.[A-Za-z]", s) is not None


def rule_19_decoy_1(s: str) -> bool:
    """
    The string is accepted when it includes an alphanumeric token that starts with an uppercase letter and ends with digits.
    """
    return re.search(r"(?<![A-Za-z0-9])[A-Z][A-Za-z0-9]*\d(?![A-Za-z0-9])", s) is not None


# def rule_19_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 19: Requires a token that has letters, then digits, then more letters (e.g., "A12B").
#     """
#     return re.search(r"(?<![A-Za-z0-9])[A-Za-z]+\d+[A-Za-z]+[A-Za-z0-9]*", s) is not None


# def rule_19_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 19: Detects tokens where digits bookend at least one alphabetic segment, like "7X2".
#     """
#     return re.search(r"(?<![A-Za-z0-9])\d+[A-Za-z]+\d+[A-Za-z0-9]*", s) is not None


def rule_20_decoy_1(s: str) -> bool:
    """
    The string is accepted when it contains an email-like token whose top-level domain may include digits.
    """
    pattern = re.compile(
        r"(?<![A-Za-z0-9._%+\-])"
        r"[A-Za-z0-9._%+\-]+"
        r"@"
        r"[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"
        r"(?![A-Za-z0-9._%+\-])"
    )
    return pattern.search(s) is not None


# def rule_20_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 20: Accepts addresses only when the domain has at least two dots (e.g., user@sub.example.com).
#     """
#     pattern = re.compile(
#         r"(?<![A-Za-z0-9._%+\-])"
#         r"[A-Za-z0-9._%+\-]+"
#         r"@"
#         r"[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+){2,}"
#         r"(?![A-Za-z0-9._%+\-])"
#     )
#     return pattern.search(s) is not None


# def rule_20_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 20: Looks for local@domain patterns but ignores the requirement that the TLD be alphabetic.
#     """
#     pattern = re.compile(
#         r"(?<![A-Za-z0-9._%+\-])"
#         r"[A-Za-z0-9._%+\-]+"
#         r"@"
#         r"[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"
#         r"(?![A-Za-z0-9._%+\-])"
#     )
#     match = pattern.search(s)
#     if not match:
#         return False
#     domain = match.group(0).split("@", 1)[1]
#     tld = domain.split(".")[-1]
#     return any(ch.isdigit() for ch in tld)


def rule_21_decoy_1(s: str) -> bool:
    """
    The string is accepted when its square brackets are balanced and at least one bracketed segment contains an uppercase letter.
    """
    stack = []
    segments = []
    for idx, ch in enumerate(s):
        if ch == "[":
            stack.append(idx)
        elif ch == "]":
            if not stack:
                return False
            start = stack.pop()
            segments.append(s[start + 1 : idx])
    if stack:
        return False
    return bool(segments) and any(any(ch.isupper() for ch in segment) for segment in segments)


# def rule_21_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 21: Accepts bracketed spans whose first character inside the brackets is a digit.
#     """
#     return re.search(r"\[[0-9][^\[\]]*\]", s) is not None


# def rule_21_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 21: Looks for bracketed segments that include a colon character.
#     """
#     return re.search(r"\[[^\[\]]*:[^\[\]]*\]", s) is not None


# def rule_22_decoy_1(s: str) -> bool:
#     """
#     Decoy 1 for Rule 22: Searches for words of length five or more whose consonants never repeat
#     (vowels may repeat).
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     for word in words:
#         lw = word.lower()
#         if len(lw) < 5:
#             continue
#         consonants = [ch for ch in lw if ch not in "aeiou"]
#         if consonants and len(set(consonants)) == len(consonants):
#             return True
#     return False


# def rule_22_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 22: Accepts words of length five or more whose vowels are all distinct.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     for word in words:
#         lw = word.lower()
#         if len(lw) < 5:
#             continue
#         vowels = [ch for ch in lw if ch in "aeiou"]
#         if len(vowels) >= 2 and len(set(vowels)) == len(vowels):
#             return True
#     return False


# def rule_22_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 22: Finds words of length three or more whose letters increase strictly in alphabetical order.
#     """
#     words = re.findall(r"[A-Za-z]+", s)
#     for word in words:
#         lw = word.lower()
#         if len(lw) < 3:
#             continue
#         if all(lw[i] < lw[i + 1] for i in range(len(lw) - 1)):
#             return True
#     return False


def rule_23_decoy_1(s: str) -> bool:
    """
    The string is accepted when it contains a 24-hour time between 06:00 and 20:59 inclusive.
    """
    pattern = r"(?<![A-Za-z0-9])(?:0[6-9]|1\d|20):[0-5]\d(?![A-Za-z0-9])"
    return re.search(pattern, s) is not None


# def rule_23_decoy_2(s: str) -> bool:
#     """
#     Decoy 2 for Rule 23: Matches times that include explicit seconds in HH:MM:SS format.
#     """
#     pattern = r"(?<![A-Za-z0-9])(?:[01]?\d|2[0-3]):[0-5]\d:[0-5]\d(?![A-Za-z0-9])"
#     return re.search(pattern, s) is not None


# def rule_23_decoy_3(s: str) -> bool:
#     """
#     Decoy 3 for Rule 23: Accepts 24-hour times using a dot as the separator, such as 18.45.
#     """
#     pattern = r"(?<![A-Za-z0-9])(?:[01]?\d|2[0-3])\.[0-5]\d(?![A-Za-z0-9])"
#     return re.search(pattern, s) is not None


def rule_24_decoy_1(s: str) -> bool:
    """
    The string is accepted when it contains a word whose letters increase strictly in alphabetical order.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for word in words:
        lw = word.lower()
        if len(lw) < 3:
            continue
        if all(lw[i] < lw[i + 1] for i in range(len(lw) - 1)):
            return True
    return False


def rule_24_decoy_2(s: str) -> bool:
    """
    The string is accepted when it contains a word whose letters are in nonincreasing alphabetical order.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for word in words:
        lw = word.lower()
        if len(lw) < 3:
            continue
        if all(lw[i] >= lw[i + 1] for i in range(len(lw) - 1)):
            return True
    return False


def rule_24_decoy_3(s: str) -> bool:
    """
    The string is accepted when it contains a three-letter word whose letters appear in nondecreasing alphabetical order.
    """
    words = re.findall(r"[A-Za-z]+", s)
    for word in words:
        lw = word.lower()
        if len(lw) != 3:
            continue
        if lw[0] <= lw[1] <= lw[2]:
            return True
    return False


def rule_25_decoy_1(s: str) -> bool:
    """
    The string is accepted when it contains a date in YYYY-MM-DD format without checking whether the day is valid for the month.
    """
    return re.search(r"(?<![A-Za-z0-9])\d{4}-\d{2}-\d{2}(?![A-Za-z0-9])", s) is not None


def rule_25_decoy_2(s: str) -> bool:
    """
    The string is accepted when it contains a year-month expression in the format YYYY-MM.
    """
    return re.search(r"(?<![A-Za-z0-9])\d{4}-(0[1-9]|1[0-2])(?![A-Za-z0-9])", s) is not None


def rule_25_decoy_3(s: str) -> bool:
    """
    The string is accepted when it contains a US-style date written as MM/DD/YYYY.
    """
    return re.search(r"(?<![A-Za-z0-9])(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}(?![A-Za-z0-9])", s) is not None
