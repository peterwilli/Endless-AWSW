import re

tokens_to_expect = {
    'cmd_scn': "<scn>".split(),
    'cmd_msg': "<msg>".split(),
    'cmd_p': "<p>".split(),
    'cmd_d': "<d>".split(),
}
allowed_commands = [
    "msg",
    "scn"
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
re_token = re.compile(r'(<.*?>|[^<]*)')
re_command = re.compile(r'^<(.*?)>$')
re_alphanumeric_whitespace = re.compile(r'[A-Za-z0-9\s]')
re_within_message = re.compile(r'[\sa-zA-Z0-\[\]\-\+]')

class ValidationException(Exception):
    pass

class ValidatedReplyBuffer:
    def __init__(self):
        self.tokens = []
        self.last_cmd = None
        self.last_side = None
        self.expect_new_tokens(tokens_to_expect['cmd_p'])
        self.in_message = False

    def expect_new_tokens(self, tokens):
        self.expect_tokens = tokens
        self.expect_tokens_idx = 0
    
    def add_token(self, token):
        expected_token = self.expect_tokens[self.expect_tokens_idx]
        if type(expected_token) == re.Pattern:
            if expected_token.match(token) is None:
                raise ValidationException(f"add_token[re]: {token} != {expected_token}!")
        else:
            if token != expected_token:
                raise ValidationException(f"add_token[str]: {token} != {expected_token}!")

        self.tokens.append(token)
        self.expect_tokens_idx += 1
        if self.expect_tokens_idx == len(self.expect_tokens):
            cmd_match = re_command.match("".join(self.expect_tokens))
            if cmd_match is not None:
                self.last_cmd = cmd_match.group(1)
                if not self.last_cmd in allowed_commands:
                    raise ValidationException(f"add_token: {self.last_cmd} is not an allowed command!")
            if self.last_cmd == 'p':
                self.expect_new_tokens(tokens_to_expect['cmd_msg'])
            elif self.last_cmd == 'd':
                self.last_side = 'd'
            elif self.last_cmd == 'msg':
                if token == ' ':
                    # with a space we check the character that came before
                    character = "".join(self.tokens[self.token_last_index(">"):])
                    if not character in allowed_characters:
                        raise ValidationException(f"add_token: character '{character}' not in allowed_characters!")
                    self.expect_new_tokens(['"'])
                elif token == '"':
                    self.in_message = not self.in_message
                    if self.is_in_message:
                        self.expect_new_tokens([re_within_message])
                    else:
                        if self.last_side == 'p':
                            self.expect_new_tokens(tokens_to_expect['cmd_d'])
                        elif self.last_side == 'd':
                            self.expect_new_tokens(tokens_to_expect['cmd_p'])
                        else:
                            raise Exception(f"invalid last side: {self.last_side} can either be d or p!")
                else:
                    self.expect_new_tokens([re_alphanumeric_whitespace])

    def token_last_index(self, token) -> int:
        return len(self.tokens) - operator.indexOf(reversed(self.tokens), token) - 1

    def squeeze(self) -> str:
        return "".join(self.tokens)