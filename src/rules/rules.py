############################################################################################################
# rules.py: All rules as functions
# each rule is numbered sequentially, the docstring explains the rule in natural language, 
# the rule takes in a string and returns True or False depending on whether the string satisifies the rule
# use make_rules.py to generate new rules.
############################################################################################################

def rule_1(s: str) -> bool:
    """
    Rule 1: Length less than or equal to 10 words

    Examples:
    >>> rule_1("Breakfast was late but the coffee was perfect.")
    True
    >>> rule_1("We took the slow train along the coast at sunset.")
    True
    >>> rule_1("Mist rolled over the fields.")
    True
    >>> rule_1("Turn left at the old stone bridge by the river.")
    True
    >>> rule_1("I wandered through the market, sampling spices and bargaining for little gifts.")
    False
    """
    length = len(s.split(" "))
    return length <= 10

def rule_2(s: str) -> bool:
    """
    Rule 2: Starts with a capital letter (first non-space character is uppercase)

    Examples:
    >>> rule_2("Autumn storms rattled the windows all night.")
    True
    >>> rule_2("   Somewhere beyond the dunes, music drifted softly.")
    True
    >>> rule_2("midnight trains rarely keep to the timetable.")
    False
    >>> rule_2("7pm dinners never start on time in this house.")
    False
    >>> rule_2('"whenever you visit, bring marmalade.')
    False
    """
    s = s.lstrip()
    return bool(s) and s[0].isalpha() and s[0].isupper()

def rule_3(s: str) -> bool:
    """
    Rule 3: The string contains an even number of decimal digits (0-9). Zero digits counts as even.
    Only characters '0' through '9' are considered; all other characters are ignored.

    Examples:
    >>> rule_3("The curtains fluttered in the evening breeze.")
    True
    >>> rule_3("The receipt shows 24 apples and 16 pears.")
    True
    >>> rule_3("Gate B3 changed to C4 just before boarding.")
    True
    >>> rule_3("Our meeting starts at 9 and ends by 12.")
    False
    >>> rule_3("We booked seats 17A and 18C for the trip.")
    True
    """
    digit_count = sum(1 for ch in s if '0' <= ch <= '9')
    return digit_count % 2 == 0

def rule_4(s: str) -> bool:
    """
    Rule 4: The string contains at least one palindromic word of length 3 or more.
    A "word" is any contiguous run of alphabetic letters (A-Z or a-z). Matching is case-insensitive.

    Examples:
    >>> rule_4("We saw a level canoe on the lake.")
    True
    >>> rule_4("Madam, your order is ready.")
    True
    >>> rule_4("We walked along the river path at dusk.")
    False
    >>> rule_4("Go to the park at noon.")
    True
    >>> rule_4("Fresh bread cooled on the kitchen table.")
    False
    """
    import re
    words = re.findall(r"[A-Za-z]+", s)
    for w in words:
        lw = w.lower()
        if len(lw) >= 3 and lw == lw[::-1]:
            return True
    return False

def rule_5(s: str) -> bool:
    """
    Rule 5: The string contains at least one word ending with 'ing' and at least one word ending with 'ed'.
    A "word" is any contiguous run of alphabetic letters (A-Z or a-z). Matching is case-insensitive.

    Examples:
    >>> rule_5("We enjoyed hiking and chatted by the fire.")
    True
    >>> rule_5("Sanded boards kept warping in the humid shed.")
    True
    >>> rule_5("They walked and laughed together through town.")
    False
    >>> rule_5("We are singing, dancing, and exploring by moonlight.")
    False
    >>> rule_5("Painted signs hung over the bustling market.")
    True
    """
    import re
    words = re.findall(r"[A-Za-z]+", s)
    has_ing = False
    has_ed = False
    for w in words:
        lw = w.lower()
        if lw.endswith("ing"):
            has_ing = True
        if lw.endswith("ed"):
            has_ed = True
        if has_ing and has_ed:
            return True
    return False

def rule_6(s: str) -> bool:
    """
    Rule 6: The string contains at least one hyphenated word where both sides are purely alphabetic
    and at least two letters long. Matching is case-insensitive. A pattern like "well-lit" qualifies,
    while "x-ray" (one-letter side) or "3-4" (digits) do not.

    Examples:
    >>> rule_6("We took a last-minute detour around the village.")
    True
    >>> rule_6("The well-lit stairway felt safe at night.")
    True
    >>> rule_6("They offered sugar-free cookies and hot tea.")
    True
    >>> rule_6("The museum gift shop sold X-ray postcards.")
    False
    >>> rule_6("Cost was 3-4 dollars during the sale.")
    False
    """
    import re
    return re.search(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b", s) is not None

def rule_7(s: str) -> bool:
    """
    Rule 7: The string contains at least one alphabetic word that includes a double letter
    (the same letter appearing twice in a row). A "word" is any contiguous run of letters
    A-Z or a-z. Matching is case-insensitive.

    Examples:
    >>> rule_7("The coffee smelled sweet at sunrise.")
    True
    >>> rule_7("We booked a small room near the harbor.")
    True
    >>> rule_7("A narrow alley led to the museum.")
    True
    >>> rule_7("The gray fog drifted over the bay.")
    False
    >>> rule_7("Bring a map and water for the hike.")
    False
    """
    import re
    words = re.findall(r"[A-Za-z]+", s)
    for w in words:
        lw = w.lower()
        for i in range(len(lw) - 1):
            if lw[i] == lw[i + 1]:
                return True
    return False

def rule_8(s: str) -> bool:
    """
    Rule 8: The string contains at least one repeated word: the same alphabetic word
    appears two or more times anywhere in the string, case-insensitive.
    A "word" is any contiguous run of letters A-Z or a-z; punctuation and digits are ignored.

    Examples:
    >>> rule_8("We will, we will rock you.")
    True
    >>> rule_8("Again and again the waves returned.")
    True
    >>> rule_8("No no, that trail loops back.")
    True
    >>> rule_8("Strangers shared stories by the fire.")
    False
    >>> rule_8("Cats, dogs, and birds watched quietly.")
    False
    """
    import re
    words = re.findall(r"[A-Za-z]+", s)
    seen = set()
    for w in words:
        lw = w.lower()
        if lw in seen:
            return True
        seen.add(lw)
    return False

def rule_9(s: str) -> bool:
    """
    Rule 9: The string contains at least one word with an internal apostrophe:
    a letter, followed by an apostrophe (straight ' or curly ’), followed by a letter.
    Examples include can't, we'll, o'clock, teacher's. Apostrophes at the very start
    or end of a word do not count. Matching is case-insensitive.

    Examples:
    >>> rule_9("We can't stay long because it's late.")
    True
    >>> rule_9("I found my neighbor's keys by the stairs.")
    True
    >>> rule_9("They will arrive at noon; the road is clear.")
    False
    >>> rule_9("'Hello' was painted on the sign outside.")
    False
    >>> rule_9("The clock struck o'clock and everyone cheered.")
    True
    """
    import re
    # Match any occurrence of Letter + (apostrophe or curly apostrophe) + Letter
    return re.search(r"[A-Za-z][\'\u2019][A-Za-z]", s) is not None

def rule_10(s: str) -> bool:
    """
    Rule 10: The string has balanced parentheses, and at least one matched pair of parentheses
    contains at least one alphabetic letter (A-Z or a-z) somewhere between the '(' and ')'.
    Parentheses are balanced if no closing ')' appears before a corresponding '(' and the total
    number of '(' equals the total number of ')'. Nested pairs are allowed.

    Examples:
    >>> rule_10("Pack the boxes (carefully) before the rain.")
    True
    >>> rule_10("They arrived (on time (mostly)) after all.")
    True
    >>> rule_10("He waved (twice and smiled.")
    False
    >>> rule_10("Notes () are incomplete.")
    False
    >>> rule_10("We timed it (3:45) exactly.")
    False
    """
    stack = []
    has_alpha_inside_any_pair = False

    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if not stack:
                return False  # Unmatched closing parenthesis
            start = stack.pop()
            segment = s[start + 1 : i]
            # Check for at least one ASCII alphabetic character inside this pair
            if any(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in segment):
                has_alpha_inside_any_pair = True

    # Balanced if no unmatched opening parentheses remain
    return (len(stack) == 0) and has_alpha_inside_any_pair

def rule_11(s: str) -> bool:
    """
    Rule 11: The string contains at least one instance of a four-letter word immediately
    followed by a five-letter word. A "word" is any contiguous run of alphabetic letters
    (A-Z or a-z). Matching is case-insensitive.

    Examples:
    >>> rule_11("This story moves quickly.")
    True
    >>> rule_11("We took a long train across town.")
    True
    >>> rule_11("Neat facts emerge early.")
    True
    >>> rule_11("The short path bends left.")
    False
    >>> rule_11("Quietly, birds sang.")
    False
    """
    import re
    words = re.findall(r"[A-Za-z]+", s)
    lengths = [len(w) for w in words]
    for i in range(len(lengths) - 1):
        if lengths[i] == 4 and lengths[i + 1] == 5:
            return True
    return False

def rule_12(s: str) -> bool:
    """
    Rule 12: The string contains at least one alphabetic word of length 4 or more
    whose letters strictly alternate between consonant and vowel (or vowel and consonant),
    across the entire word. Matching is case-insensitive, and 'y' counts as a consonant.
    A "word" is any contiguous run of letters A-Z or a-z.

    Examples:
    >>> rule_12("Paper boats drifted past the pier.")
    True
    >>> rule_12("A tiger padded silently through ferns.")
    True
    >>> rule_12("We unpacked ripe bananas for snacks.")
    True
    >>> rule_12("Quiet voices echoed in the hall.")
    False
    >>> rule_12("Strong winds shook oak trees.")
    False
    """
    import re

    def is_vowel(ch: str) -> bool:
        return ch in "aeiouAEIOU"

    words = re.findall(r"[A-Za-z]+", s)
    for w in words:
        if len(w) < 4:
            continue
        ok = True
        for i in range(1, len(w)):
            if is_vowel(w[i]) == is_vowel(w[i - 1]):
                ok = False
                break
        if ok:
            return True
    return False

def rule_13(s: str) -> bool:
    """
    Rule 13: The first and last alphabetic words in the string start with the same letter,
    case-insensitive. A "word" is any contiguous run of letters A-Z or a-z. If there are
    no alphabetic words, the rule is not satisfied.

    Examples:
    >>> rule_13("Misty hills meet midnight.")
    True
    >>> rule_13("Bright lanterns flickered by the shore.")
    False
    >>> rule_13("Well-lit walkway to the woods.")
    True
    >>> rule_13("Game at 7pm ends now.")
    False
    >>> rule_13("Orbit.")
    True
    """
    import re
    words = re.findall(r"[A-Za-z]+", s)
    if not words:
        return False
    return words[0][0].lower() == words[-1][0].lower()

def rule_14(s: str) -> bool:
    """
    Rule 14: The string contains balanced double quotes (an even count of the " character),
    and at least one quoted segment (the text between a matched pair of double quotes)
    contains a space character. Quotes are paired from left to right, and only the straight
    ASCII double quote character (") is considered.

    Examples:
    >>> rule_14('We waited as the announcement said "final call for boarding" just before noon.')
    True
    >>> rule_14('She whispered, "good night, everyone," and closed the door.')
    True
    >>> rule_14('The label read "Fragile." on the box.')
    False
    >>> rule_14('The chalkboard says "Back soon.')
    False
    >>> rule_14('Lights dimmed as the curtains closed.')
    False
    """
    quote_indices = [i for i, ch in enumerate(s) if ch == '"']
    if len(quote_indices) < 2 or len(quote_indices) % 2 != 0:
        return False

    for i in range(0, len(quote_indices), 2):
        start = quote_indices[i]
        end = quote_indices[i + 1]
        segment = s[start + 1 : end]
        if " " in segment:
            return True
    return False

def rule_15(s: str) -> bool:
    """
    Rule 15: The string contains at least one comma (,) and at least one period (.).
    Commas and periods may appear anywhere in the string.

    Examples:
    >>> rule_15("We packed, then we left.")
    True
    >>> rule_15("Yes, we can. Let's go")
    True
    >>> rule_15("Please close the door.")
    False
    >>> rule_15("Wait, please")
    False
    >>> rule_15("Sunrise over hills")
    False
    """
    return ("," in s) and ("." in s)

def rule_16(s: str) -> bool:
    """
    Rule 16: The string contains exactly one question mark ('?') and no exclamation marks ('!').
    Other characters and punctuation are allowed and ignored.

    Examples:
    >>> rule_16("Are we meeting at noon?")
    True
    >>> rule_16("Is this the right door? He knocked twice.")
    True
    >>> rule_16("Tell me your answer.")
    False
    >>> rule_16("Where did you go? Did you see the lights?")
    False
    >>> rule_16("What a surprise! You arrived early?")
    False
    """
    return s.count("?") == 1 and "!" not in s

def rule_17(s: str) -> bool:
    """
    Rule 17: The string contains at least one integer written with comma thousands separators.
    A valid instance is a run of 1–3 digits followed by one or more groups of a comma and exactly
    three digits (e.g., 1,000; 12,345; 2,147,483,648). The matched number must not be immediately
    followed by a digit or by a decimal fraction (a dot and a digit).

    Examples:
    >>> rule_17("Attendance reached 12,350 today.")
    True
    >>> rule_17("We sold 1,000 tickets in two hours.")
    True
    >>> rule_17("Price is 12,345.67 now, so we waited.")
    False
    >>> rule_17("The old map lists 1,234,567; navigation was easy.")
    True
    >>> rule_17("The budget was 1000 dollars this quarter.")
    False
    """
    import re
    pattern = r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)(?!\.\d)"
    return re.search(pattern, s) is not None

def rule_18(s: str) -> bool:
    """
    Rule 18: The string contains at least one ellipsis of exactly three consecutive periods ("...").
    The ellipsis must not be part of a longer run of periods; sequences of two dots or four or more
    dots do not qualify.

    Examples:
    >>> rule_18("We waited... then we left.")
    True
    >>> rule_18("Well... okay... fine.")
    True
    >>> rule_18("He paused.... then spoke.")
    False
    >>> rule_18("I counted.. not enough.")
    False
    >>> rule_18("Just wait.")
    False
    """
    import re
    return re.search(r"(?<!\.)\.\.\.(?!\.)", s) is not None

def rule_19(s: str) -> bool:
    """
    Rule 19: The string contains at least one alphanumeric token that mixes letters and digits
    within a single contiguous run. A token is any contiguous sequence of ASCII letters and digits
    (A–Z, a–z, 0–9). The token must include at least one letter and at least one digit.

    Examples:
    >>> rule_19("We parked at Apt4B near the alley.")
    True
    >>> rule_19("Her code name was x3 but nobody used it.")
    True
    >>> rule_19("Please wait in room 12B by the elevators.")
    True
    >>> rule_19("We counted 42 boxes on the shelf.")
    False
    >>> rule_19("Quiet birds drifted over the bay.")
    False
    """
    import re
    tokens = re.findall(r"[A-Za-z0-9]+", s)
    for t in tokens:
        has_letter = any(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in t)
        has_digit = any('0' <= c <= '9' for c in t)
        if has_letter and has_digit:
            return True
    return False

def rule_20(s: str) -> bool:
    """
    Rule 20: The string contains at least one token that looks like a valid email address.
    A valid instance has:
      - a local part of one or more characters from letters, digits, '.', '_', '%', '+', or '-'
      - a single '@' symbol
      - a domain made of two or more labels separated by dots; each label has letters, digits, or hyphens,
        and the final label (top-level domain) is letters only with length at least 2
      - the address is not embedded inside a longer sequence of the allowed local/domain characters

    Examples:
    >>> rule_20("Please send your notes to team.leads+west@dept.example.org by Friday.")
    True
    >>> rule_20("Reach me at A_B9@mail.co.uk, and copy the lead.")
    True
    >>> rule_20("The RSVP address (info@studio-77.com) is in the invite.")
    True
    >>> rule_20("He wrote user@localhost on the board.")
    False
    >>> rule_20("They scribbled work.id@10.0.0.5, then crossed it out.")
    False
    """
    import re

    # Match candidate emails with boundaries that exclude being part of a larger token
    pattern = re.compile(
        r"(?<![A-Za-z0-9._%+\-])"                 # left boundary: not allowed local/domain char
        r"([A-Za-z0-9._%+\-]+)"                    # local part
        r"@"
        r"([A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+)"   # domain with at least one dot
        r"(?![A-Za-z0-9._%+\-])"                   # right boundary: not allowed local/domain char
    )

    for m in pattern.finditer(s):
        domain = m.group(2)
        labels = domain.split(".")
        # Final label must be all letters and at least 2 characters
        tld = labels[-1]
        if not (tld.isalpha() and len(tld) >= 2):
            continue
        # No label may start or end with a hyphen, and labels must be non-empty
        if any((not lab) or lab[0] == "-" or lab[-1] == "-" for lab in labels):
            continue
        return True
    return False

def rule_21(s: str) -> bool:
    """
    Rule 21: The string has balanced square brackets ('[' and ']'), and at least one matched pair
    of brackets contains at least one decimal digit (0-9) somewhere between the '[' and ']'.
    Brackets are balanced if no closing ']' appears before a corresponding '[' and the total
    number of '[' equals the total number of ']'. Nested pairs are allowed.

    Examples:
    >>> rule_21("Please tag [item 7] before shipment.")
    True
    >>> rule_21("Nested sets like [A[2]B] are fine.")
    True
    >>> rule_21("We visited [room 12] upstairs.")
    True
    >>> rule_21("He wrote [notes] and left.")
    False
    >>> rule_21("Pack the boxes [carefully.")
    False
    """
    stack = []
    has_digit_inside_any_pair = False

    for i, ch in enumerate(s):
        if ch == '[':
            stack.append(i)
        elif ch == ']':
            if not stack:
                return False  # Unmatched closing bracket
            start = stack.pop()
            segment = s[start + 1 : i]
            if any('0' <= c <= '9' for c in segment):
                has_digit_inside_any_pair = True

    return (len(stack) == 0) and has_digit_inside_any_pair

def rule_22(s: str) -> bool:
    """
    Rule 22: The string contains at least one alphabetic word of length 5 or more
    in which no letter repeats. A "word" is any contiguous run of letters (A-Z or a-z).
    Matching is case-insensitive.

    Examples:
    >>> rule_22("Crisp blank pages waited in the journal.")
    True
    >>> rule_22("We chose planet as the project name.")
    True
    >>> rule_22("Ivory teacups gleam beside the shelf.")
    True
    >>> rule_22("The coffee smelled sweet.")
    False
    >>> rule_22("Hannah added pepper to the soup.")
    False
    """
    import re
    words = re.findall(r"[A-Za-z]+", s)
    for w in words:
        lw = w.lower()
        if len(lw) >= 5 and len(set(lw)) == len(lw):
            return True
    return False

def rule_23(s: str) -> bool:
    """
    Rule 23: The string contains at least one valid 24-hour clock time in the exact form HH:MM,
    where HH is 00–23 and MM is 00–59. The time must not be directly preceded or followed by
    an alphanumeric character (to avoid being part of a longer token like 1207:45 or 07:45pm).
    Matching is case-insensitive and considers only ASCII digits.

    Examples:
    >>> rule_23("The first train leaves at 07:45 from platform 2.")
    True
    >>> rule_23("We will check in around 19:00; dinner follows.")
    True
    >>> rule_23("They arrived at 9:05 sharp after the rain.")
    False
    >>> rule_23("The log shows 23:75 due to a typo.")
    False
    >>> rule_23("Backup started at 24:00 on the server.")
    False
    """
    import re
    pattern = re.compile(r"(?<![A-Za-z0-9])(?:[01]\d|2[0-3]):[0-5]\d(?![A-Za-z0-9])")
    return pattern.search(s) is not None

def rule_24(s: str) -> bool:
    """
    Rule 24: The string contains at least one alphabetic word of length 3 or more
    whose letters are in nondecreasing alphabetical order (each letter is the same as
    or comes after the previous letter in the alphabet). Matching is case-insensitive.
    A "word" is any contiguous run of letters A-Z or a-z.

    Examples:
    >>> rule_24("We chose to act now.")
    True
    >>> rule_24("The door stood open at dawn.")
    True
    >>> rule_24("Aegilops appears in some lists.")
    True
    >>> rule_24("Brisk winds shook the tents.")
    False
    >>> rule_24("Keep calm and carry on.")
    False
    """
    import re
    words = re.findall(r"[A-Za-z]+", s)
    for w in words:
        lw = w.lower()
        if len(lw) < 3:
            continue
        ok = True
        for i in range(1, len(lw)):
            if lw[i] < lw[i - 1]:
                ok = False
                break
        if ok:
            return True
    return False

def rule_25(s: str) -> bool:
    """
    Rule 25: The string contains at least one valid ISO calendar date in the exact form YYYY-MM-DD,
    where YYYY is 1000–2999, MM is 01–12, and DD is a valid day for that month (including leap-year
    handling for February). The date must not be directly preceded or followed by an alphanumeric
    character (A–Z, a–z, 0–9).

    Examples:
    >>> rule_25("The deadline is 2025-11-30 for all teams.")
    True
    >>> rule_25("We met on 2000-02-29 to celebrate.")
    True
    >>> rule_25("Record the event (2019-07-04) in the log.")
    True
    >>> rule_25("That typo 2021-02-29 slipped into the report.")
    False
    >>> rule_25("Use 2023-12-01a as a placeholder code.")
    False
    """
    import re

    pattern = re.compile(
        r"(?<![A-Za-z0-9])"            # not directly preceded by alphanumeric
        r"([12]\d{3})"                 # year: 1000-2999
        r"-(0[1-9]|1[0-2])"            # month: 01-12
        r"-(0[1-9]|[12]\d|3[01])"      # day: 01-31 (validated further below)
        r"(?![A-Za-z0-9])"             # not directly followed by alphanumeric
    )

    for m in pattern.finditer(s):
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))

        # Determine max day for the month, considering leap years
        if month in (1, 3, 5, 7, 8, 10, 12):
            max_day = 31
        elif month in (4, 6, 9, 11):
            max_day = 30
        else:  # February
            is_leap = (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))
            max_day = 29 if is_leap else 28

        if 1 <= day <= max_day:
            return True

    return False
