def normalize_response(text: str) -> str:
    """Normalize the response text by removing extra spaces and converting to lowercase"""
    return ' '.join(text.lower().split())

def parse_likert_response(raw_response: str) -> str:
    """Parse the raw response to extract the standardized Likert scale response"""
    response_mapping = {
        "very much like me": [
            "very much like me",
            "likes me very much",
            "like me very much"
        ],
        "like me": [
            "like me",
            "likes me"
        ],
        "somewhat like me": [
            "somewhat like me",
            "somewhat likes me",
            "some what like me",
            "some what likes me",
        ],
        "a little like me": [
            "a little like me",
            "little like me",
        ],
        "not like me": [
            "not like me",
            "does not like me",
            "doesn't like me",
            "is not like me"
        ],
        "not like me at all": [
            "not like me at all",
            "not at all like me",
            "does not like me at all",
            "isn't like me at all"
        ]
    }
    
    normalized_response = normalize_response(raw_response)
    
    # First, try exact matches
    for standard_form, variations in response_mapping.items():
        for variant in variations:
            if normalize_response(variant) == normalized_response:
                return standard_form

    # If no exact match, try "contains matching" and collect all matches
    possible_matches = []
    for standard_form, variations in response_mapping.items():
        for variant in variations:
            n_variant = normalize_response(variant)
            if n_variant in normalized_response:
                possible_matches.append((standard_form, n_variant))
    
    # If we have multiple possible matches, choose the one with the longest matching variant
    if possible_matches:
        possible_matches.sort(key=lambda x: len(x[1]), reverse=True)
        return possible_matches[0][0]
            
    raise ValueError(f"Could not parse valid Likert response from: {raw_response}")

def map_response_to_numeric(parsed_response: str) -> int:
    """Map the parsed Likert response to a numeric value from 6 to 1"""
    response_scores = {
        "very much like me": 6,
        "like me": 5,
        "somewhat like me": 4,
        "a little like me": 3,
        "not like me": 2,
        "not like me at all": 1
    }
    return response_scores[parsed_response]