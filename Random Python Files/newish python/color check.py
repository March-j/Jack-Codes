import re
VALID_COLORS = ['blue', 'red']
HEX_COLOR_REGEX = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'

def is_hex_color(input_string):
    regexp = re.compile(HEX_COLOR_REGEX)
    if regexp.search(input_string):
        return True
    return False

def is_color(input_string):
    for color in VALID_COLORS:
        if color in input_string or is_hex_color(input_string):
            return True
    return False
