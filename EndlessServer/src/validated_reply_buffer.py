import re
import operator
import logging

tokens_to_expect = {
    'cmd_scn': "<scn>",
    'cmd_msg': "<msg>",
    'cmd_p': "<p>",
    "cmd_p_or_d": ['<', re.compile(r'p|d'), '>'],
    'cmd_d': "<d>",
}
for k in tokens_to_expect:
    tokens_to_expect[k] = list(tokens_to_expect[k])
allowed_commands = [
    "msg",
    "scn",
    "p",
    "d"
]
allowed_characters = {
    'c': 'Player',
    'Ry': 'Remy',
    'Lo': 'Lorem',
    'Ip': 'Ipsum',
    'Br': 'Bryce',
    'Wr': 'Unknown name',
    'Em': 'Emera',
    'Ka': 'Katsuharu',
    'Rz': 'Reza',
    'Kv': 'Kevin',
    'Zh': 'Zhong',
    'm': 'Narrator',
    'n': 'Back Story',
    'Mv': 'Maverick',
    'An': 'Anna',
    'Ad': 'Adine',
    'Sb': 'Sebastian'
}
allowed_scenes = ['park2', 'black', 'loremapt', 'office', 'bare', 'bareblur', 'bareblur2', 'pad', 'facin2', 'facinx', 'facin3', 'alley', 'farm', 'town4', 'beach', 'adineapt', 'corridor', 'emeraroom', 'o4', 'park3', 'np3x', 'np2x', 'np1x', 'buildingoutside', 'o2', 'np3', 'np2', 'store2', 'town1x', 'forestx', 'cave', 'o', 'remyapt', 'cafe', 'viewingspot', 'np1r', 'hallway', 'np2y', 'np1n', 'town2', 'stairs', 'darker', 'town1', 'store', 'library', 'school', 'forest1', 'forest2', 'storex', 'np5e', 'port1', 'beachx', 'padx', 'intro1', 'intro2', 'np4', 'np5', 'fac1', 'facin', 'town3', 'kitchen', 'np1', 'stars', 'o3', 'town7', 'town6', 'deadbody', 'whiteroom', 'office2', 'cave2', 'table', 'starsrx', 'hatchery', 'farm2', 'gate', 'testingroom', 'np6', 'fac12', 'adineapt2']        

re_token = re.compile(r'(<.*?>|[^<]*)')
re_command = re.compile(r'^<(.*?)>$')
re_alphanumeric_whitespace = re.compile(r'[A-Za-z0-9\s]')
re_alphanumeric_scn = re.compile(r'[a-z0-9<>]')
re_within_message = re.compile(r'[\sa-zA-Z0-\[\]\-\+\?\"\.!\',]')
re_brackets = re.compile(r'\[(.*?)]')

class ValidationException(Exception):
    pass

def has_unclosed_or_nested_brackets(text) -> bool:
    is_ok = True
    for char in text:
        if char == '[':
            if is_ok:
                is_ok = False
            else:
                return True
        elif char == ']':
            if is_ok:
                return True
            else:
                is_ok = True
    return not is_ok

def has_valid_bracket_vars(text) -> bool:
    valid_var_names = ['player_name']

    for var_name in re_brackets.findall(text):
        if var_name not in valid_var_names:
            return False
            
    return True

class ValidatedReplyBuffer:
    def __init__(self, initial_state: str = None):
        self.tokens = ""
        self.last_cmd = None
        self.last_side = None
        self.last_character = None
        self.expect_new_tokens(tokens_to_expect['cmd_p'])
        self.in_message = False
        if initial_state is not None:
            for t in initial_state:
                self.add_token(t, False)

    def expect_new_tokens(self, tokens, index_override = 0):
        self.expect_tokens = tokens
        self.expect_tokens_idx = index_override
    
    def add_token(self, token: str, is_computer_generated: bool) -> int:
        expect_tokens_len = len(self.expect_tokens)
        if self.expect_tokens_idx >= expect_tokens_len:
            raise Exception(f"expect_tokens_idx({self.expect_tokens_idx}) > expect_tokens {self.expect_tokens} ({len(self.expect_tokens)})")
        expected_token = self.expect_tokens[self.expect_tokens_idx]
        if type(expected_token) == re.Pattern:
            if expected_token.match(token) is None:
                raise ValidationException(f"add_token[re]: expected '{expected_token}', got '{token}' (hex: {ord(token)})! (full text so far: {self.tokens})")
        else:
            if token != expected_token:
                raise ValidationException(f"add_token[str]: expected '{expected_token}', got '{token}' (hex: {ord(token)})! (full text so far: {self.tokens})")

        self.tokens += token
        self.expect_tokens_idx += 1
        if self.expect_tokens_idx == expect_tokens_len:
            cmd_match = re_command.match("".join(self.tokens[expect_tokens_len * -1:]))
            if cmd_match is not None:
                self.last_cmd = cmd_match.group(1)
                if not self.last_cmd in allowed_commands:
                    raise ValidationException(f"add_token: {self.last_cmd} is not an allowed command!")
            if self.last_cmd == 'p':
                self.last_side = 'p'
                self.expect_new_tokens(tokens_to_expect['cmd_msg'])
            elif self.last_cmd == 'd':
                self.last_side = 'd'
                self.expect_new_tokens(tokens_to_expect['cmd_scn'])
            elif self.last_cmd == 'scn':
                if token == '<':
                    # The scene ended
                    scene = "".join(self.tokens[self.tokens_last_index(">") + 1:len(self.tokens) - 1])
                    if not scene in allowed_scenes:
                        raise ValidationException(f"add_token: scene '{scene}' is not in allowed_scenes!")
                    # We already have the <
                    self.expect_new_tokens(tokens_to_expect['cmd_msg'], index_override = 1)
                else:
                    self.expect_new_tokens([re_alphanumeric_scn])
            elif self.last_cmd == 'msg':
                if token == '"':
                    self.in_message = not self.in_message
                    if self.in_message:
                        self.expect_new_tokens([re_within_message])
                    else:
                        new_message = self.tokens[self.tokens_last_index("<msg>"):-1]
                        if not has_valid_bracket_vars(new_message):
                            raise ValidationException(f"add_token: new_message has invalid bracket vars! (Message: '{new_message}')")
                        if has_unclosed_or_nested_brackets(new_message):
                            raise ValidationException(f"add_token: new_message has unclosed or nested brackets! (Message: '{new_message}')")
                        if self.last_side == 'p':
                            self.expect_new_tokens(tokens_to_expect['cmd_d'])
                        elif self.last_side == 'd':
                            self.expect_new_tokens(tokens_to_expect['cmd_p_or_d'])
                        else:
                            raise Exception(f"invalid last side: {self.last_side} can either be d or p!")
                elif self.in_message:
                    self.expect_new_tokens([re_within_message])
                else:
                    if token == ' ':
                        # with a space we check the character that came before
                        character = ''.join(self.tokens[self.tokens_last_index('>') + 1:len(self.tokens) - 1])
                        if character == self.last_character:
                            # We don't allow the same dragon to reply twice.
                            self.tokens = self.tokens[:self.tokens_last_index("<d>")]
                            return 1
                        if is_computer_generated and character == 'c':
                            raise ValidationException("AI cannot respond as player!")
                        if not character in allowed_characters:
                            raise ValidationException(f"add_token: character '{character}' not in allowed_characters!")
                        self.last_character = character
                        self.expect_new_tokens(['"'])
                    else:
                        self.expect_new_tokens([re_alphanumeric_whitespace])
        return 0

    def tokens_last_index(self, tokens: str) -> int:
        return self.tokens.rfind(tokens)

if __name__ == '__main__':
    def test_tokens(tokens):
        print(f"testing: {tokens}")
        buffer = ValidatedReplyBuffer()
        for t in tokens:
            buffer.add_token(t)
        assert buffer.squeeze() == tokens

    test_tokens('<p><msg>c "Flooding?"<d><scn>o2<msg>Sb "Yes."')
    try:
        test_tokens('<p><msgf>c "Flooding?"<d><scn>o2<msg>Sb "Yes."')
    except ValidationException as e:
        print(e)
    try:
        test_tokens('<p><msg>c "Floodi"ng?"<d><scn>o2<msg>Sb "Yes."')
    except ValidationException as e:
        print(e)
    test_tokens('<p><msg>c "Hey Remy!"')
    test_tokens('<p><msg>c "Hey Remy!"<d><scn>o2<msg>Ry "Are you the Ghoster?"<d><scn>o2<msg>Sb "Yes."')