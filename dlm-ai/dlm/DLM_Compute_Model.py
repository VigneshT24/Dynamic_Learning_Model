import difflib
import random
import re
import nltk
from word2number import w2n

def set_geometric_height(tokens, lower_tokens) -> list | None:
    """
    Extracts the height value and its relative index from a tokenized query.

    Args:
        tokens (list): Original token list from the query.
        lower_tokens (list): Lowercased version of tokens for matching.

    Returns:
        list or None: [height_value, height_value_index] if found, otherwise None.
    """
    return_list = []
    height_value = None
    height_value_index = None
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
    if (height_value is not None and height_value_index is not None):
        return_list.append(height_value)
        return_list.append(height_value_index)
        return return_list
    else:
        return None

def set_other_geometric_values(tokens, height_value_index) -> list:
    """
    Extracts all numeric values from tokens, excluding the identified height value.

    Args:
        tokens (list): Token list from the query.
        height_value_index (int): Index of the height value to skip.

    Returns:
        list: All other numeric values found as floats.
    """
    other_values = []
    for i, token in enumerate(tokens):
        if i == height_value_index:
            continue  # skip the height value itself
        try:
            num = w2n.word_to_num(token)
            other_values.append(num)
        except ValueError:
            if token.replace('.', '', 1).isdigit():
                other_values.append(float(token))
    return other_values

def set_geometric_object_intel(self, lower_tokens) -> list:
    """
    Identifies the geometric shape and the target calculation (e.g., area, volume).

    Uses fuzzy matching on unigrams and bigrams to detect shapes and operations, handling common suffix variations (e.g., '-ular', '-ish').

    Args:
        lower_tokens (list): Lowercased token list from the query.

    Returns:
        list: Keywords describing the operation and shape (e.g., ['area', 'triangle']).
    """
    object_intel = []
    common_endings = ["ular", "ish", "al"]  # some people might say "squarish" or "rectangular" etc
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
    return object_intel

def display_geometric_inner_thought(object_intel, display_thought, height_value, other_values) -> str:
    """
    Prints the bot's internal reasoning for geometric calculations.

    Args:
        object_intel (list): List containing the calculation type and shape name.
        display_thought (bool): Flag to enable console output.
        height_value (float or None): The extracted height dimension.
        other_values (list): Additional numeric dimensions.

    Returns:
        str: The name of the identified geometric shape.
    """
    obj_name = object_intel[1]
    if display_thought:
        print(f"It seems that the user wants to compute the {' of a '.join(object_intel)}")
        if height_value is not None:
            print(f"* The user has mentioned that the height of the {obj_name} object is {height_value}")
        else:
            print(f"* The {object_intel[1]} object has no height associated with it, so moving on")
        if len(other_values) > 0:
            print(f"* Additional numerical values associated with the dimensions of the {obj_name} object is {' and '.join(str(v) for v in other_values)}")
        else:
            print(f"* No additional numerical values associated with the dimensions of the {obj_name} were given")
    return obj_name

def compute_geometrically(self, obj_name, height_value, other_values, display_thought, object_intel) -> float | None:
    """
    Calculates the geometric result using the identified shape's specific formula.

    Args:
        obj_name (str): Name of the geometric shape.
        height_value (float or None): Height dimension.
        other_values (list): Additional numeric dimensions.
        display_thought (bool): Flag to enable error message output.
        object_intel (list): Calculation type and shape for contextual error reporting.

    Returns:
        float or None: The calculated result rounded to 4 decimals, or None if inputs mismatch the formula.
    """
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
                f"Unable to compute the {object_intel[0]} of the {obj_name} due to missing or mismatched values")
        else:
            print(
                f"Unable to compute the {object_intel[0]} of the {obj_name} due to missing or mismatched values")
        return None

def geometric_calculation(self, filtered_query, display_thought) -> float | None:  
    """
    Orchestrates the full geometric calculation pipeline.

    Extracts dimensions, identifies the shape/operation, logs the thought process, and computes the final result.

    Args:
        filtered_query (str): The cleaned user query containing computational details.
        display_thought (bool): Flag to enable Chain-of-Thought console output.

    Returns:
        float or None: The computed geometric result, or None if it fails.
    """
    tokens = filtered_query.split()
    lower_tokens = [t.lower() for t in tokens]

    # extract height
    new_height_list = set_geometric_height(tokens, lower_tokens)
    if new_height_list is not None:
        height_value = new_height_list[0]
        height_value_index = new_height_list[1]
    else:
        height_value = None
        height_value_index = None

    # extract other values
    other_values = set_other_geometric_values(tokens, height_value_index)

    # identify object
    object_intel = set_geometric_object_intel(self, lower_tokens)

    # display thought process
    obj_name = display_geometric_inner_thought(object_intel, display_thought, height_value, other_values)

    # compute and return result
    return compute_geometrically(self, obj_name, height_value, other_values, display_thought, object_intel)

def initialize_cot_variables() -> tuple:
    """
    Initializes data structures and keyword lists required for Chain-of-Thought reasoning.

    Returns:
        tuple: (persons_mentioned, keywords_mentioned, num_mentioned, operands_mentioned, arithmetic_ending_phrases)
    """
    persons_mentioned = []
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
    return persons_mentioned, keywords_mentioned, num_mentioned, operands_mentioned, arithmetic_ending_phrases

def pick_out_names(self, doc, persons_mentioned, filtered_query) -> set:
    """
    Extracts proper nouns and recognized names from the query using SpaCy and NLTK.

    Args:
        doc (spacy.tokens.Doc): Initialized SpaCy NLP document of the query.
        persons_mentioned (list): Buffer list to append identified person names.
        filtered_query (str): The cleaned user query.

    Returns:
        set: A collection of all non-person proper nouns (items) extracted.
    """
    # Have the bot pick out names mentioned (in order) using SpaCy and NLTK (for maximum coverage)
    items_mentioned = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            cleaned = re.sub(r'\d+', "", ent.text).strip()
            if cleaned:
                persons_mentioned.append(cleaned)

    # this is to solve a problematic issue with nltk if a certain package is not already downloaded
    try:
        tokens = nltk.word_tokenize(filtered_query)
    except LookupError:
        # silently download the tokenizer if it is missing
        nltk.download('punkt_tab', quiet=True)
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
    return items_mentioned

def display_initial_thought(display_thought, filtered_query) -> None:
    """
    Prints the introductory Chain-of-Thought message.

    Args:
        display_thought (bool): Flag to enable console output.
        filtered_query (str): The stripped query being analyzed.
    """
    if display_thought:
        print(f"I am presented with a more involved query asking me to do some form of computation")
        print("Let me think about this carefully and break it down so that I can solve it")
        print(f"I've trimmed away any extra words so I'm focusing on \"{filtered_query}\" now")

def check_if_geometric_query(self, filtered_query, display_thought) -> tuple:
    """
    Determines if the query is geometric and attempts to solve it if applicable.

    Args:
        filtered_query (str): The cleaned user query.
        display_thought (bool): Flag to enable Chain-of-Thought console output.

    Returns:
        tuple: (is_geometric_query (bool), geometric_ans (float or None)).
    """
    words = filtered_query.lower().split()
    geometric_ans = None

    geometric_calc = any(
        difflib.get_close_matches(word, self._DLM__geometric_calculation_identifiers.keys(), n=1, cutoff=0.70)
        for word in words
    )
    is_geometric_query = False

    geo_types = set()
    for t in self._DLM__geometric_calculation_identifiers:
        shape = self._DLM__geometric_calculation_identifiers[t]["keywords"]
        geo_types.add(shape[0])

    if any(difflib.get_close_matches(word, geo_types, n=1, cutoff=0.70) for word in words) and geometric_calc:
        geometric_ans = geometric_calculation(self, filtered_query, display_thought)
        if geometric_ans is not None:
            is_geometric_query = True

    return is_geometric_query, geometric_ans

def extract_operands(self, filtered_query, arithmetic_ending_phrases, keywords_mentioned, operands_mentioned) -> None:
    """
    Extracts arithmetic operands (e.g., +, -) by matching keywords and symbols in the query.

    Modifies the keywords_mentioned and operands_mentioned lists in-place.

    Args:
        filtered_query (str): The cleaned user query.
        arithmetic_ending_phrases (list): Common concluding phrases to ignore or use for context.
        keywords_mentioned (list): Buffer to append found operational keywords.
        operands_mentioned (list): Buffer to append the mapped mathematical operators.
    """
    tokens_lower = filtered_query.lower().split()
    last_two = set(tokens_lower[-2:])

    found_operand = False
    for fq in filtered_query.split():
        fq_l = fq.lower()

        if fq_l in arithmetic_ending_phrases and fq_l in last_two:
            continue
        if fq_l in {"+", "-", "*", "/"}:
            operands_mentioned.append(fq_l)
            keywords_mentioned.append(fq_l)
            continue

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
                            break
                        continue
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

def extract_operands_from_ending_phrases(self, filtered_query, arithmetic_ending_phrases, keywords_mentioned, operands_mentioned) -> None:
    """
    Acts as a fallback to extract mathematical operands based on concluding sentence phrases.

    Modifies the keywords_mentioned and operands_mentioned lists in-place.

    Args:
        filtered_query (str): The cleaned user query.
        arithmetic_ending_phrases (list): Contextual phrases indicating specific operations.
        keywords_mentioned (list): Buffer to append found operational keywords.
        operands_mentioned (list): Buffer to append the mapped mathematical operators.
    """
    if not operands_mentioned:
        for fq in filtered_query.split():
            p_fq = self._DLM__nlp(fq)

            matched_ep = None
            for ep in arithmetic_ending_phrases:
                p_ep = self._DLM__nlp(ep)
                if p_ep.vector_norm != 0 and p_fq.vector_norm != 0 and p_ep.similarity(p_fq) > 0.50:
                    matched_ep = ep
                    break

            if not matched_ep:
                continue

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

def extract_numbers(self, filtered_query, operands_mentioned, num_mentioned) -> None:
    """
    Extracts all numeric values from the query, converting text representations (e.g., 'half', 'triple') to floats.

    Modifies the num_mentioned list in-place.

    Args:
        filtered_query (str): The cleaned user query.
        operands_mentioned (list): Currently identified mathematical operators.
        num_mentioned (list): Buffer to append the extracted numerical values as strings.
    """
    text_nums = ["a", "an", "half", "double", "triple", "quadruple"]
    a_an_detected = False

    tokens = filtered_query.lower().split()
    for token in tokens:
        if re.fullmatch(r"\d+(\.\d+)?", token):
            num_mentioned.append(str(float(token)))
            continue

        try:
            num = w2n.word_to_num(token)
            num_mentioned.append(str(float(num)))
            continue
        except ValueError:
            pass

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

    if a_an_detected and (num_mentioned.count("1.0") > 1 or len(num_mentioned) > 1):
        num_mentioned.remove("1.0")

def handle_equals_operand(operands_mentioned, num_mentioned) -> None:
    """
    Adjusts the operands list for unit conversions or isolated equals signs.

    Modifies the operands_mentioned list in-place.

    Args:
        operands_mentioned (list): Current list of mathematical operators.
        num_mentioned (list): Current list of extracted numbers.
    """
    if ('=' in operands_mentioned) and (len(num_mentioned) < 2):
        operands_mentioned.clear()
        operands_mentioned.append('=')
    else:
        if '=' in operands_mentioned:
            operands_mentioned[:] = [op for op in operands_mentioned if op != '=']

def check_missing_components(self, is_geometric_query, num_mentioned, operands_mentioned, display_thought) -> bool:
    """
    Validates whether enough numeric and operational components exist to perform a calculation.

    Args:
        is_geometric_query (bool): Flag indicating if the query was successfully handled geometrically.
        num_mentioned (list): Extracted numerical values.
        operands_mentioned (list): Extracted mathematical operators.
        display_thought (bool): Flag to enable console output.

    Returns:
        bool: True if essential components are missing (computation should abort), False otherwise.
    """
    if (not is_geometric_query) and (any(not lst for lst in (num_mentioned, operands_mentioned)) or (
            '=' not in operands_mentioned and num_mentioned.__len__() < 2)):
        if (not self._DLM__try_compute):
            if display_thought:
                print(
                    f"{'Hmm...' or '' if display_thought else ''}It looks like some essential details are missing, so I can't complete this calculation right now.")
            self._DLM__try_memory = True
        else:
            print("Hmm...")
        return True
    return False

def display_extracted_components(display_thought, persons_mentioned, items_mentioned, is_geometric_query, num_mentioned, keywords_mentioned, operands_mentioned) -> None:
    """
    Prints the compiled list of entities, numbers, and operators found during analysis.

    Args:
        display_thought (bool): Flag to enable console output.
        persons_mentioned (set): Identified names.
        items_mentioned (set): Identified objects.
        is_geometric_query (bool): Flag indicating if the query is geometric.
        num_mentioned (list): Extracted numerical values.
        keywords_mentioned (list): Extracted operation keywords.
        operands_mentioned (list): Extracted mathematical operators.
    """
    if display_thought:
        print(
            f"1.) I see {', '.join(persons_mentioned) if persons_mentioned.__len__() >= 1 else 'no one'} mentioned as a person name; "
            f"{"they're likely key to this problem" if persons_mentioned.__len__() >= 1 else 'moving on'}")
        print(f"2.) Moreover, I see {', '.join(items_mentioned) if items_mentioned.__len__() >= 1 else 'no items'} mentioned as proper nouns; "
            f"{'this might be a key thing to this problem' if items_mentioned.__len__() >= 1 else 'moving on'}")
        if is_geometric_query:
            print(f"3.) This is a geometric problem and I have already computed the answer")
        else:
            print(f"3.) I've also identified the numbers {' and '.join(num_mentioned)} that I need to compute with")
            print(
                f"4.) I see the keywords \"{'\" and \"'.join(keywords_mentioned)}\", meaning I need to perform a \"{'\" and \"'.join(operands_mentioned)}\" operation for this query; I'll use that to guide my calculation")
            print("Now I have the parts, so let me put it all together and solve")

def reorder_numbers_by_indicators(filtered_query, num_mentioned) -> None:
    """
    Reorders extracted numbers, moving the 'original' or 'initial' value to the front of the list.

    Modifies the num_mentioned list in-place.

    Args:
        filtered_query (str): The cleaned user query.
        num_mentioned (list): The list of extracted numerical string values.
    """
    indicators = {"original", "originally", "initial", "initially", "at first", "to begin with", "had",
                  "savings", "saving", "of"}

    tokens = filtered_query.split()
    temp = None
    lower_tokens = [t.lower() for t in tokens]

    for idx, token in enumerate(lower_tokens):
        if token in indicators:
            if idx > 0 and token != "of":
                candidate = lower_tokens[idx - 1]
                try:
                    temp = (w2n.word_to_num(candidate))
                except ValueError:
                    pass
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

def compute_geometric_problem(self, geometric_ans) -> None:
    """
    Finalizes and logs a successful geometric computation.

    Args:
        geometric_ans (float): The previously calculated geometric result.
    """
    print(f"Geometric Answer: {geometric_ans}")
    self._DLM__successfully_computed = True

def compute_conversion_problem(self, filtered_query, num_mentioned, display_thought) -> None:
    """
    Executes a unit conversion calculation by identifying source and target units in the query.

    Args:
        filtered_query (str): The cleaned user query.
        num_mentioned (list): Extracted numerical values.
        display_thought (bool): Flag to enable Chain-of-Thought console output.
    """
    try:
        tokens = filtered_query.lower().split()
        num0 = float(num_mentioned[0])
        num_idx = None

        text_nums = {
            "a": 1.0,
            "an": 1.0,
            "half": 0.5,
            "double": 2.0,
            "triple": 3.0,
            "quadruple": 4.0
        }

        for i, tok in enumerate(tokens):
            lower_tok = tok.lower()

            if lower_tok in text_nums:
                if text_nums[lower_tok] == num0:
                    num_idx = i
                    break
                else:
                    continue

            try:
                if float(tok) == num0:
                    num_idx = i
                    break
            except ValueError:
                try:
                    if float(w2n.word_to_num(tok)) == num0:
                        num_idx = i
                        break
                except ValueError:
                    continue

        source_key = None
        target_key = None

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

        if source_key and target_key:
            result = (num0 * self._DLM__units[source_key]) / self._DLM__units[target_key]
            if display_thought:
                print(
                    f"I need to take {num0} and multiply it by {self._DLM__units[source_key]}. Finally, I divide by {self._DLM__units[target_key]} and I got my answer")
            expr = f"{num_mentioned[0]} {source_key}(s) ==> {round(result, 2)} {target_key}(s)"
            print(f"Conversion Answer: {expr}")
            self._DLM__successfully_computed = True
        else:
            print(f"Could not identify both source and target units.")
    except SyntaxError:
        print("Oops! I still mix up conversions and arithmetic sometimes. Working on it!")

def compute_arithmetic_problem(self, filtered_query, num_mentioned, operands_mentioned) -> None:
    """
    Constructs and evaluates a standard mathematical expression from extracted numbers and operands.

    Args:
        filtered_query (str): The cleaned user query.
        num_mentioned (list): Extracted numerical values.
        operands_mentioned (list): Extracted mathematical operators.
    """
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
        print(f"Arithmetic Answer: {expr} = {result}")
        self._DLM__successfully_computed = True
    except SyntaxError:
        print(f"Something about that stumped me. I'll need to learn more to handle it properly.")

def handle_computation_failure(self, keywords_mentioned) -> None:
    """
    Logs a failure message when the computation engine lacks the context to solve the problem.

    Args:
        keywords_mentioned (list): Extracted operation keywords to display in the error message.
    """
    self._DLM__successfully_computed = False
    print(f"{random.choice(self._DLM__fallback_responses)}")
    print(
        f"However, while I was trying to understand the math, I ran into \"{'" and "'.join(keywords_mentioned)}\", which I use to connect keywords to math operations.")
    print(f"That might've confused me a bit, maybe try leaving one of those out or rephrase it to make it clearer?")

def perform_advanced_CoT(self, filtered_query, display_thought) -> None:
    """
    Orchestrates the entire Chain-of-Thought reasoning process to solve mathematical word problems.

    This pipeline extracts entities, maps textual phrases to mathematical operations, isolates numerical 
    values, and routes the data to geometric, conversion, or arithmetic sub-engines. It directly handles 
    console output for the reasoning process based on the display flag.

    Args:
        filtered_query (str): The cleaned, logic-based user query.
        display_thought (bool): Flag to enable step-by-step reasoning output to the console.
    """
    # Initialize variables
    persons_mentioned, keywords_mentioned, num_mentioned, operands_mentioned, arithmetic_ending_phrases = initialize_cot_variables()

    filtered_query = filtered_query.title()
    doc = self._DLM__nlp(filtered_query)

    # Display initial thought
    display_initial_thought(display_thought, filtered_query)

    # Extract person names and items
    items_mentioned = pick_out_names(self, doc, persons_mentioned, filtered_query)

    # Check if geometric query
    is_geometric_query, geometric_ans = check_if_geometric_query(self, filtered_query, display_thought)

    # If not geometric, extract operands and numbers
    if not is_geometric_query:
        extract_operands(self, filtered_query, arithmetic_ending_phrases, keywords_mentioned, operands_mentioned)
        extract_operands_from_ending_phrases(self, filtered_query, arithmetic_ending_phrases, keywords_mentioned, operands_mentioned)
        keywords_mentioned[:] = list(dict.fromkeys(keywords_mentioned))
        extract_numbers(self, filtered_query, operands_mentioned, num_mentioned)
        handle_equals_operand(operands_mentioned, num_mentioned)

    # Check if components are missing
    if check_missing_components(self, is_geometric_query, num_mentioned, operands_mentioned, display_thought):
        return

    # Display extracted components
    display_extracted_components(display_thought, persons_mentioned, items_mentioned, is_geometric_query, num_mentioned,
                                 keywords_mentioned, operands_mentioned)

    # Reorder numbers by indicators
    reorder_numbers_by_indicators(filtered_query, num_mentioned)

    # Compute based on problem type
    if is_geometric_query:
        compute_geometric_problem(self, geometric_ans)
    elif len(num_mentioned) == 1 and len(operands_mentioned) == 1:
        compute_conversion_problem(self, filtered_query, num_mentioned, operands_mentioned, display_thought)
    elif len(num_mentioned) >= 2 and (
            len(operands_mentioned) == (len(num_mentioned) - 1) or len(operands_mentioned) == 1):
        compute_arithmetic_problem(self, filtered_query, num_mentioned, operands_mentioned)
    else:
        handle_computation_failure(self, keywords_mentioned)