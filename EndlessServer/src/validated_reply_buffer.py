import re
import operator

tokens_to_expect = {
    'cmd_scn': "<scn>",
    'cmd_msg': "<msg>",
    'cmd_p': "<p>",
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
re_within_message = re.compile(r'[\sa-zA-Z0-\[\]\-\+\?\"\.]')

class ValidationException(Exception):
    pass

class ValidatedReplyBuffer:
    def __init__(self):
        self.tokens = []
        self.last_cmd = None
        self.last_side = None
        self.expect_new_tokens(tokens_to_expect['cmd_p'])
        self.in_message = False

    def expect_new_tokens(self, tokens, index_override = 0):
        self.expect_tokens = tokens
        self.expect_tokens_idx = index_override
    
    def add_token(self, token):
        if self.expect_tokens_idx >= len(self.expect_tokens):
            raise Exception(f"expect_tokens_idx({self.expect_tokens_idx}) > expect_tokens {self.expect_tokens} ({len(self.expect_tokens)})")
        expected_token = self.expect_tokens[self.expect_tokens_idx]
        if type(expected_token) == re.Pattern:
            if expected_token.match(token) is None:
                raise ValidationException(f"add_token[re]: expected '{expected_token}', got '{token}'!")
        else:
            if token != expected_token:
                raise ValidationException(f"add_token[str]: expected '{expected_token}', got '{token}'!")

        self.tokens.append(token)
        self.expect_tokens_idx += 1
        if self.expect_tokens_idx == len(self.expect_tokens):
            try_cmd_match = True
            for t in self.expect_tokens:
                if type(t) == re.Pattern:
                    try_cmd_match = False
                    break
            if try_cmd_match:
                cmd_match = re_command.match("".join(self.expect_tokens))
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
                    scene = "".join(self.tokens[self.token_last_index(">") + 1:len(self.tokens) - 1])
                    if not scene in allowed_scenes:
                        raise ValidationException(f"add_token: scene '{scene}' is not in allowed_scenes!")
                    # We already have the <
                    self.expect_new_tokens(tokens_to_expect['cmd_msg'], index_override = 1)
                else:
                    self.expect_new_tokens([re_alphanumeric_scn])
            elif self.last_cmd == 'msg':
                if token == ' ':
                    # with a space we check the character that came before
                    character = "".join(self.tokens[self.token_last_index(">") + 1:len(self.tokens) - 1])
                    if not character in allowed_characters:
                        raise ValidationException(f"add_token: character '{character}' not in allowed_characters!")
                    self.expect_new_tokens(['"'])
                elif token == '"':
                    self.in_message = not self.in_message
                    if self.in_message:
                        self.expect_new_tokens([re_within_message])
                    else:
                        if self.last_side == 'p':
                            self.expect_new_tokens(tokens_to_expect['cmd_d'])
                        elif self.last_side == 'd':
                            self.expect_new_tokens(tokens_to_expect['cmd_p'])
                        else:
                            raise Exception(f"invalid last side: {self.last_side} can either be d or p!")
                elif self.in_message:
                    self.expect_new_tokens([re_within_message])
                else:
                    self.expect_new_tokens([re_alphanumeric_whitespace])

    def token_last_index(self, token) -> int:
        return len(self.tokens) - operator.indexOf(reversed(self.tokens), token) - 1

    def squeeze(self) -> str:
        return "".join(self.tokens)

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
    