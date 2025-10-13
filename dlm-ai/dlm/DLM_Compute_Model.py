import difflib
import random
import re
import nltk
from word2number import w2n

def geometric_calculation(self, filtered_query, display_thought):  # returns float result or None
    """
    Perform geometric problems that will be called inside perform_advanced_CoT.

    Parameters:
        filtered_query (str): user query that has been filtered to have mostly computational details.
        display_thought (bool): Indicates whether the user wants to have the bot display its thought process or just give the answer.

    Returns:
        float: The result after computing the geometric calculation.

    Behavior:
        - Search through query to find specific keywords like 'area' or 'volume'.
        - Then, search to find shape or object to perform math on like 'triangle' or 'square'.
        - Find numbers associated with object details and store in appropriate list.
        - Finally, find appropriate formula with identifiers and plug in and return answer.

    """
    height_value = None
    height_value_index = None
    other_values = []
    object_intel = []
    common_endings = ["ular", "ish", "al"]  # some people might say "squarish" or "rectangular" etc

    tokens = filtered_query.split()
    lower_tokens = [t.lower() for t in tokens]

    # set height value number (if exists) to height_value and also its corresponding index
    for idx, token in enumerate(lower_tokens):
        is_similar = difflib.get_close_matches(token, ["height"], n=1, cutoff=0.7)
        if is_similar and is_similar[0] == "height":
            # try token before
            if idx > 0:
                candidate = lower_tokens[idx - 1]
                try:
                    height_value = w2n.word_to_num(candidate)
                    height_value_index = idx - 1
                except ValueError:
                    if candidate.replace('.', '', 1).isdigit():
                        height_value = float(candidate)
                        height_value_index = idx - 1
                        break
                    pass

            # try token after
            if idx < len(tokens) - 1:
                candidate = lower_tokens[idx + 1]
                try:
                    height_value = w2n.word_to_num(candidate)
                    height_value_index = idx + 1
                except ValueError:
                    if candidate.replace('.', '', 1).isdigit():
                        height_value = float(candidate)
                        height_value_index = idx + 1
                        break
                    pass

    # append all other numbers to "other_value" list
    for i, token in enumerate(tokens):
        if i == height_value_index:
            continue  # skip the height value itself
        try:
            num = w2n.word_to_num(token)
            other_values.append(num)
        except ValueError:
            if token.replace('.', '', 1).isdigit():
                other_values.append(float(token))

    # find the object name and what to compute about the object
    bigrams = [" ".join([lower_tokens[i], lower_tokens[i + 1]]) for i in range(len(lower_tokens) - 1)]
    end_check = False

    # first check bi-grams
    for phrase in bigrams:
        for obj in self._DLM__geometric_calculation_identifiers:
            for ending in common_endings:
                if phrase[0].endswith(ending):
                    phrase = phrase[: -len(ending)]
                    break
            is_similar = difflib.get_close_matches(phrase, [obj], n=1, cutoff=0.70)
            if is_similar and is_similar[0] == obj:
                geom_type = self._DLM__geometric_calculation_identifiers[obj]["keywords"]
                if (lower_tokens.__contains__(geom_type[0])):
                    object_intel.extend(geom_type)
                    end_check = True
                    break
                else:
                    continue
        if end_check:
            break

    # if no bi-gram match, check single words
    if not end_check and not lower_tokens.__contains__("prism"):
        for token in lower_tokens:
            for obj in self._DLM__geometric_calculation_identifiers:
                for ending in common_endings:
                    if token.endswith(ending):
                        token = token[: -len(ending)]
                        break
                is_similar = difflib.get_close_matches(token, [obj], n=1, cutoff=0.80)
                if is_similar and is_similar[0] == obj:
                    object_intel.extend(self._DLM__geometric_calculation_identifiers[obj]["keywords"])
                    end_check = True
                    break
            if end_check:
                break

    # if allowed, display the inner thought process
    obj_name = object_intel[1]
    if display_thought:
        self._DLM__loading_animation(f"It seems that the user wants to compute the {' of a '.join(object_intel)}", 0.5)
        if height_value is not None:
            self._DLM__loading_animation(
                f"* The user has mentioned that the height of the {obj_name} object is {height_value}", 0.4)
        else:
            self._DLM__loading_animation(
                f"* The {object_intel[1]} object has no height associated with it, so moving on", 0.4)
        if len(other_values) > 0:
            self._DLM__loading_animation(
                f"* Additional numerical values associated with the dimensions of the {obj_name} object is {' and '.join(str(v) for v in other_values)}",
                0.4)
        else:
            self._DLM__loading_animation(
                f"* No additional numerical values associated with the dimensions of the {obj_name} were given",
                0.4)

    # Now iterate through the geometric identifier list, find the correct object, and then find its formula, then plug compute
    formula = self._DLM__geometric_calculation_identifiers[obj_name]["formula"]
    params = self._DLM__geometric_calculation_identifiers[obj_name]["params"]

    formula_inputs = {}  # all data gathered to compute geometry

    # gather and plug in values into the formula
    try:
        if "height" in params:
            formula_inputs["height"] = height_value
        if "side" in params:
            formula_inputs["side"] = height_value

        value_idx = 0  # count how many values to be added in formula_inputs
        for param in params:
            if len(other_values) < 1:
                break
            if param == "height":
                continue  # already added
            elif param == "other":  # two consecutive numbers to append
                formula_inputs["other"] = other_values[value_idx:value_idx + 2]
                value_idx += 2
            else:  # only one number to append
                formula_inputs[param] = other_values[value_idx]
                value_idx += 1
                if len(other_values) <= 1:
                    break

        if "height" in params:
            if formula_inputs["height"] is None and len(other_values) > 1:
                formula_inputs["height"] = other_values[len(other_values) - 1]
                other_values.pop(len(other_values) - 1)

        # Try calculating the result and return
        result = round(formula(formula_inputs), 4)
        return result

    except Exception as e:
        if display_thought:
            print(
                f"{'\033[33m'}Unable to compute the {object_intel[0]} of the {obj_name} due to missing or mismatched values{'\033[0m'}")
        else:
            print(
                f"{'\033[34m'}Unable to compute the {object_intel[0]} of the {obj_name} due to missing or mismatched values{'\033[0m'}")
        return None


def perform_advanced_CoT(self, filtered_query, display_thought):  # no return, void
    """
    Perform advanced Chain-of-Thought (CoT) reasoning to solve arithmetic or unit conversion problems.

    Parameters:
        filtered_query (str): The cleaned user input, expected to be a math or logic-based question.
        display_thought (bool): Indicates whether the user wants to have the bot display its thought process or just give the answer.

    Behavior:
        - Simulates step-by-step reasoning to solve arithmetic word problems without relying on memorized answers.
        - Extracts entities including person names, items, numbers, and operations using SpaCy, NLTK, and regex.
        - Detects arithmetic operations via lexical and semantic matching with predefined keyword sets.
        - Handles both numeric digits and text-based numbers (e.g., "three", "double").
        - Supports simple arithmetic expressions and unit conversions (e.g., inches to cm).
        - Prints the interpreted steps, logical inferences (if display_thought is True), and the final computed result with contextual explanations.
        - Displays fallback messages if the query is incomplete or too ambiguous to solve.
    """
    persons_mentioned = []
    items_mentioned = []
    keywords_mentioned = []
    num_mentioned = []
    operands_mentioned = []
    arithmetic_ending_phrases = [
        "total", "all", "left", "leftover", "remaining", "altogether", "together", "each", "spend", "per",
        "sum", "combined", "add up", "accumulate", "bring to", "rise by", "grow by", "earned", "in all", "in total",
        "difference", "deduct", "decrease by", "fell by", "drop by", "ate",
        "multiply", "times", "product", "received", "pick", "paid", "gave", "pay",
        "split", "shared equally", "equal parts", "equal groups", "ratio", "quotient", "out of", "into"
    ]
    filtered_query = filtered_query.title()
    doc = self._DLM__nlp(filtered_query)

    if display_thought:
        print(
            f"{'\033[33m'}I am presented with a more involved query asking me to do some form of computation{'\033[0m'}")
        self._DLM__loading_animation("Let me think about this carefully and break it down so that I can solve it", 0.8)
        self._DLM__loading_animation(f"I’ve trimmed away any extra words so I’m focusing on \"{filtered_query}\" now",
                                 0.8)

    # Have the bot pick out names mentioned (in order) using SpaCy and NLTK (for maximum coverage)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            cleaned = re.sub(r'\d+', "", ent.text).strip()
            if cleaned:
                persons_mentioned.append(cleaned)

    tokens = nltk.word_tokenize(filtered_query)
    for tok in tokens:
        cleaned = re.sub(r"[^a-zA-Z]", "", tok).lower()
        if cleaned in self._DLM__nltk_names:
            persons_mentioned.append(cleaned.capitalize())
    persons_mentioned = {name for name in set(persons_mentioned) if len(name.split()) == 1}
    persons_mentioned = set(persons_mentioned)

    # Have the bot pick out item names (in order) using SpaCy
    for token in doc:
        if token.pos_ == "PROPN":
            cleaned = re.sub(r'\d+', "", token.text).strip()
            if cleaned and cleaned not in persons_mentioned:
                items_mentioned.append(cleaned)
    items_mentioned = set(items_mentioned)

    tokens_lower = filtered_query.lower().split()
    last_two = set(tokens_lower[-2:])  # only the final 2 words from filtered input

    # First see if the problem is a geometric problem
    words = filtered_query.lower().split()
    geometric_ans = None
    # checks if the query contains shapes or object to perform possibly formula calculation
    geometric_calc = any(
        difflib.get_close_matches(word, self._DLM__geometric_calculation_identifiers.keys(), n=1, cutoff=0.70) for word
        in words)
    is_geometric_query = False

    geo_types = set()  # currently supported types of geometric calculations

    for t in self._DLM__geometric_calculation_identifiers:
        shape = self._DLM__geometric_calculation_identifiers[t]["keywords"]
        geo_types.add(shape[0])
    if any(difflib.get_close_matches(word, geo_types, n=1, cutoff=0.70) for word in words) and geometric_calc:
        geometric_ans = geometric_calculation(self, filtered_query, display_thought)
        if geometric_ans is not None:
            is_geometric_query = True
    else:  # Not geometric, so have the bot find all operand indicating keywords
        found_operand = False
        for fq in filtered_query.split():
            fq_l = fq.lower()
            # If this word is one of the ending phrases and sits among the last five, skip it
            if fq_l in arithmetic_ending_phrases and fq_l in last_two:
                continue
            if fq_l in {"+", "-", "*", "/"}:
                operands_mentioned.append(fq_l)
                keywords_mentioned.append(fq_l)
                continue  # move on to the next token
            for operand, keywords in self._DLM__computation_identifiers.items():
                for kw in keywords:
                    p1 = self._DLM__nlp(kw)
                    p2 = self._DLM__nlp(fq_l)
                    word_num_surrounded = re.search(rf'\d+\s*{fq.lower()}\s*\d+', filtered_query.lower())

                    # Direct match or lemma match
                    if (kw.lower() == fq.lower()) or p1[0].lemma_ == p2[0].lemma_:
                        keywords_mentioned.append(kw.title())
                        if kw.lower() == "out of":
                            if word_num_surrounded:
                                operands_mentioned.append(operand)
                                found_operand = True
                                break  # only break if 'out of' condition is satisfied
                            continue  # skip adding 'out of' if not surrounded by numbers
                        else:
                            operands_mentioned.append(operand)
                            found_operand = True
                            break

                    # Vector + string similarity
                    if p1.vector_norm != 0 and p2.vector_norm != 0 and (
                            p1.similarity(p2) > 0.80 and difflib.SequenceMatcher(None, kw, fq_l).ratio() > 0.40):
                        keywords_mentioned.append(kw.title())
                        if kw.lower() == "out of":
                            if word_num_surrounded:
                                operands_mentioned.append(operand)
                                found_operand = True
                                break
                            continue
                        else:
                            operands_mentioned.append(operand)
                            found_operand = True
                            break

                    # Fallback: high string similarity
                    elif difflib.SequenceMatcher(None, kw, fq_l).ratio() > 0.80:
                        keywords_mentioned.append(kw.title())
                        if kw.lower() == "out of":
                            if word_num_surrounded:
                                operands_mentioned.append(operand)
                                found_operand = True
                                break
                            continue
                        else:
                            operands_mentioned.append(operand)
                            found_operand = True
                            break

                if found_operand:
                    found_operand = False
                    break

        # If no operands were found in the main pass, check ending phrases as a last resort
        if not operands_mentioned:
            for fq in filtered_query.split():
                p_fq = self._DLM__nlp(fq)

                # Replace exact matching with a spaCy similarity check against ending_phrases
                matched_ep = None
                for ep in arithmetic_ending_phrases:
                    p_ep = self._DLM__nlp(ep)
                    if p_ep.vector_norm != 0 and p_fq.vector_norm != 0 and p_ep.similarity(p_fq) > 0.50:
                        matched_ep = ep
                        break

                if not matched_ep:
                    continue

                # Now matched_ep roughly corresponds to an ending phrase; find its operand via spaCy
                for operand, keywords in self._DLM__computation_identifiers.items():
                    for kw in keywords:
                        p_kw = self._DLM__nlp(kw)
                        if p_kw.vector_norm != 0 and p_fq.vector_norm != 0 and p_kw.similarity(p_fq) > 0.70:
                            keywords_mentioned.append(kw.title())
                            operands_mentioned.append(operand)
                            break
                    if operands_mentioned:
                        break
                if operands_mentioned:
                    break
        keywords_mentioned = list(dict.fromkeys(keywords_mentioned))

        # Now have the bot pick out numbers (in order)
        # additionally, "double", "triple", "quadruple", "half", "an" and "a" also count as numbers, in addition to text numbers (e.g. "three")
        text_nums = ["a", "an", "half", "double", "triple", "quadruple"]
        a_an_detected = False

        # Combined regex and word match pass
        tokens = filtered_query.lower().split()
        for token in tokens:
            # Check if it's a digit-based number (e.g. 600, 20.5)
            if re.fullmatch(r"\d+(\.\d+)?", token):
                num_mentioned.append(str(float(token)))
                continue

            # Check if it's a word-based number (e.g. 'three', 'double', 'a')
            try:
                num = w2n.word_to_num(token)
                num_mentioned.append(str(float(num)))
                continue
            except ValueError:
                pass  # not a word2num-recognized word

            # Check if it's in our custom list (a, an, half, double, etc.)
            for t in text_nums:
                p1 = self._DLM__nlp(token)
                p2 = self._DLM__nlp(t)
                if p1[0].lemma_ == p2[0].lemma_:
                    if t == "double":
                        num_mentioned.append(float(2).__str__())
                    elif t == "triple":
                        num_mentioned.append(float(3).__str__())
                    elif t == "half":
                        num_mentioned.append(float(0.5).__str__())
                    elif ("=" in operands_mentioned) and (t == "a" or t == "an"):
                        a_an_detected = True
                        num_mentioned.append(float(1.0).__str__())
                    elif t == "quadruple":
                        num_mentioned.append(float(4).__str__())

        # Remove "1.0" if 'a'/'an' was used in an invalid context (like not following "=")
        if a_an_detected and (num_mentioned.count("1.0") > 1 or len(num_mentioned) > 1):
            num_mentioned.remove("1.0")

        if ('=' in operands_mentioned) and (len(num_mentioned) < 2):
            operands_mentioned.clear()
            operands_mentioned.append('=')
        else:
            if '=' in operands_mentioned:
                operands_mentioned = [op for op in operands_mentioned if op != '=']

    # verify and possibly print thoughts
    if (not is_geometric_query) and (any(not lst for lst in (num_mentioned, operands_mentioned)) or (
            '=' not in operands_mentioned and num_mentioned.__len__() < 2)):  # don't compute if parts are missing
        if (not self._DLM__try_compute):
            if display_thought:
                print(f"{self._DLM__loading_animation('Hmm', 0.8) or '' if display_thought else ''}{'\033[33m'}It looks like some essential details are missing, so I can’t complete this calculation right now.{'\033[0m'}")
            self._DLM__try_memory = True
        else:
            self._DLM__loading_animation('Hmm', 0.8)
    else:  # else, the bot needs to explain what it has tokenized
        if display_thought:
            self._DLM__loading_animation(
                f"1.) I see {', '.join(persons_mentioned) if persons_mentioned.__len__() >= 1 else 'no one'} mentioned as a person name; "
                f"{'they’re likely key to this problem' if persons_mentioned.__len__() >= 1 else 'moving on'}", 0.2)
            self._DLM__loading_animation(
                f"2.) Moreover, I see {', '.join(items_mentioned) if items_mentioned.__len__() >= 1 else 'no items'} mentioned as proper nouns; "
                f"{'this might be a key thing to this problem' if items_mentioned.__len__() >= 1 else 'moving on'}",
                0.2)
            if is_geometric_query:
                self._DLM__loading_animation(f"3.) This is a geometric problem and I have already computed the answer",
                                         0.2)
            else:
                self._DLM__loading_animation(
                    f"3.) I’ve also identified the numbers {' and '.join(num_mentioned)} that I need to compute with",
                    0.2)
                self._DLM__loading_animation(
                    f"4.) I see the keywords \"{'\" and \"'.join(keywords_mentioned)}\", meaning I need to perform a \"{'\" and \"'.join(operands_mentioned)}\" operation for this query; I’ll use that to guide my calculation",
                    0.2)
                self._DLM__loading_animation("Now I have the parts, so let me put it all together and solve", 0.3)

        # Finally compute it and then give the response (if there is any)
        # move "originally" numbers to the front
        indicators = {"original", "originally", "initial", "initially", "at first", "to begin with", "had",
                      "savings", "saving", "of"}

        tokens = filtered_query.split()
        temp = None
        # lowercase copy for matching
        lower_tokens = [t.lower() for t in tokens]

        for idx, token in enumerate(lower_tokens):
            if token in indicators:
                # check token before
                if idx > 0 and token != "of":
                    candidate = lower_tokens[idx - 1]
                    try:
                        temp = (w2n.word_to_num(candidate))
                    except ValueError:
                        pass
                # check token after
                if idx < len(tokens) - 1:
                    candidate = lower_tokens[idx + 1]
                    try:
                        temp = (w2n.word_to_num(candidate))
                    except ValueError:
                        pass
        if temp is not None:
            if str(float(temp)) in num_mentioned:
                num_mentioned.remove(str(float(temp)))
            num_mentioned.insert(0, str(float(temp)))

        # geometric problem
        if is_geometric_query:
            print(f"{'\033[34m'}Geometric Answer: {geometric_ans}{'\033[0m'}")
            self._DLM__successfully_computed = True
        # conversion problem
        elif len(num_mentioned) == 1 and len(operands_mentioned) == 1:
            try:
                tokens = filtered_query.lower().split()
                num0 = float(num_mentioned[0])
                num_idx = None

                # redefine text_nums to be a dictionary instead
                text_nums = {
                    "a": 1.0,
                    "an": 1.0,
                    "half": 0.5,
                    "double": 2.0,
                    "triple": 3.0,
                    "quadruple": 4.0
                }

                # Find index of the numeric token (either digit or w2n‐convertible)
                for i, tok in enumerate(tokens):
                    lower_tok = tok.lower()

                    # Check if tok is one of the special words
                    if lower_tok in text_nums:
                        if text_nums[lower_tok] == num0:
                            num_idx = i
                            break
                        else:
                            continue  # skip parsing this token further

                    # Otherwise try parsing as a standard float
                    try:
                        if float(tok) == num0:
                            num_idx = i
                            break
                    except ValueError:
                        # If that fails, try converting via w2n.word_to_num
                        try:
                            if float(w2n.word_to_num(tok)) == num0:
                                num_idx = i
                                break
                        except ValueError:
                            continue

                source_key = None
                target_key = None

                # Look for the first unit immediately after the number for source key
                if num_idx is not None:
                    for tok in tokens[num_idx + 1:]:
                        for key, val in self._DLM__units.items():
                            p1 = self._DLM__nlp(tok)
                            p2 = self._DLM__nlp(key)
                            if p1[0].lemma_ == p2[0].lemma_:
                                source_key = key
                                break
                        if source_key:
                            break

                # 3) Now scan the entire sentence for the target-key
                for tok in tokens:
                    for key, val in self._DLM__units.items():
                        p1 = self._DLM__nlp(tok)
                        p2 = self._DLM__nlp(key)
                        p3 = self._DLM__nlp(source_key)
                        if (p1[0].lemma_ == p2[0].lemma_) and (p2[0].lemma_ != p3[0].lemma_):
                            target_key = key
                            break
                    if target_key:
                        break

                # 4) Compute only if we have both source_key and target_key
                if source_key and target_key:
                    result = (num0 * self._DLM__units[source_key]) / self._DLM__units[target_key]
                    if display_thought:
                        self._DLM__loading_animation(
                            f"I need to take {num0} and multiply it by {self._DLM__units[source_key]}. Finally, I divide by {self._DLM__units[target_key]} and I got my answer",
                            0.2)
                    expr = f"{num_mentioned[0]} {source_key}(s) ==> {round(result, 2)} {target_key}(s)"
                    print(f"{'\033[34m'}Conversion Answer: {expr} {'\033[0m'}")
                    self._DLM__successfully_computed = True
                else:
                    print(f"{'\033[33m'}Could not identify both source and target units.{'\033[0m'}")
            except SyntaxError:
                print("\033[33mOops! I still mix up conversions and arithmetic sometimes. Working on it!\033[0m")
        # regular arithmetic operations
        elif len(num_mentioned) >= 2 and (
                len(operands_mentioned) == (len(num_mentioned) - 1) or len(operands_mentioned) == 1):
            # Build a string like "n0 op0 n1 op1 n2 ... op_{N-2} n_{N-1}"
            parts = []
            for i, num in enumerate(num_mentioned):
                parts.append(str(num))
                if i < (len(num_mentioned) - 1) and ("average" in filtered_query.lower()):
                    parts.append("+")
                elif i < (len(num_mentioned) - 1) and (len(operands_mentioned) == 1):
                    parts.append(operands_mentioned[0])
                elif i < len(operands_mentioned):
                    parts.append(operands_mentioned[i])
            expr = " ".join(parts)

            try:
                result = eval(expr)
                if "average" in filtered_query.lower():
                    expr = "(" + expr + ") / " + str(len(num_mentioned))
                    result /= len(num_mentioned)
                print(f"{'\033[34m'}Arithmetic Answer: {expr} = {result}{'\033[0m'}")
                self._DLM__successfully_computed = True
            except SyntaxError:
                print(
                    f"{'\033[34mAh'}, something about that stumped me. I’ll need to learn more to handle it properly.{'\033[0m'}")
        else:
            self._DLM__successfully_computed = False
            print(f"{'\033[34m'}{random.choice(self._DLM__fallback_responses)}{'\033[0m'}")
            print(
                f"{'\033[34m'}However, while I was trying to understand the math, I ran into \"{'" and "'.join(keywords_mentioned)}\", which I use to connect keywords to math operations.{'\033[0m'}")
            print(
                f"{'\033[34m'}That might've confused me a bit, maybe try leaving one of those out or rephrase it to make it clearer?{'\033[0m'}")
