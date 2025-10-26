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
