import random
import string
import re
import urllib2, urllib
import json
import ssl
import random
import renpy

class EAWSWDebugLogger:
    def __init__(self, debug_mode):
        self.debug_mode = debug_mode
    
    def log(self, msg):
        if self.debug_mode:
            with open("eawsw_logs.log", "a") as f:
                f.write(msg + "\n")

class EAWSWClient:
    def __init__(self, hosts, mods = []):
        # Jina is currently providing EAWSW hosting for free on their JCloud service.
        # At the moment, there's no need for community services.
        self.hosts = hosts
        self.save_past_amount = 6
        self.mods = mods
        self.init_mapping()
        self.state = {
            'did_run_start_narrative': False,
            'endless_awsw_past': [],
            'start_scene': None
        }
        self.last_character = None
        self.debug_logger = EAWSWDebugLogger(False)

    def init_mapping(self):
        self.character_mapping = {
            "Ad": "adine",
            "An": "anna",
            "Br": "bryce",
            "Dm": "damion",
            "Em": "emera",
            "Ip": "ipsum",
            "Iz": "izumi",
            "Ka": "katsu",
            "Kv": "kevin",
            "Lo": "lorem",
            "Mv": "maverick",
            "Nm": "naomi",
            "Ry": "remy",
            "Rz": "reza",
            "Sb": "sebastian",
            "Zh": "zhong"
        }
        self.talk_functions = {
            "Ad": renpy.store.Ad,
            "An": renpy.store.An,
            "Br": renpy.store.Br,
            "Dm": renpy.store.Dm,
            "Em": renpy.store.Em,
            "Ip": renpy.store.Ip,
            "Iz": renpy.store.Iz,
            "Ka": renpy.store.Ka,
            "Kv": renpy.store.Kv,
            "Lo": renpy.store.Lo,
            "Mv": renpy.store.Mv,
            "Ry": renpy.store.Ry,
            "Rz": renpy.store.Rz,
            "Sb": renpy.store.Sb,
            "Zh": renpy.store.Zh
        }
        if 'naomi' in self.mods:
            self.talk_functions['Nm'] = renpy.store.Nm
        # Add self
        self.talk_functions.update({
            'm': renpy.store.m,
            'c': renpy.store.c
        })
        
    def set_start_narrative(self, start_scene, start_narrative):
        self.state['start_scene'] = start_scene
        self.state['start_narrative'] = self.eawsw_json_to_doc_array(start_narrative)
        self.last_scene = self.state['start_scene']

    def strip_past(self):
        self.state['endless_awsw_past'] = self.state['endless_awsw_past'][self.save_past_amount * -1:]
        potential_stray_dragon = self.state['endless_awsw_past'][0]
        if potential_stray_dragon['tags']['cmd'] == 'msg' and potential_stray_dragon['tags']['from'] != 'c':
            # Dragon without a scene, forbidden so we delete it too
            self.state['endless_awsw_past'].pop(0)
                
    def execute_commands(self, docs):
        for item in docs:
            cmd = item['tags']['cmd']
            if cmd == "scn":
                self.last_scene = item['text']
            elif cmd == "msg":
                msg_from = item['tags']['from']
                msg = item['text']
                emotion = None
                if 'emotion' in item['tags']:
                    emotion = item['tags']['emotion']
                if msg_from in self.character_mapping:
                    if self.character_mapping[msg_from] is None:
                        self.last_character = None
                    else:
                        if emotion is None:
                            self.last_character = '%s normal b' % (self.character_mapping[msg_from])
                        else:
                            self.last_character = '%s %s b' % (self.character_mapping[msg_from], emotion)
                else:
                    self.last_character = None
                renpy.exports.scene()
                if self.last_scene is not None:
                    renpy.exports.show(self.last_scene)
                if self.last_character is not None:
                    renpy.exports.show(self.last_character)
                if msg_from in self.talk_functions:
                    talk_fn = self.talk_functions[msg_from]
                    talk_fn(msg)     

    # Since we can't use Jina's DocumentArray directly in this mod
    # We use a sexy json and convert it to DocumentArray's json format
    def eawsw_json_to_doc_array(self, arr):
        result = []
        for cmd in arr:
            obj = {
                'tags': {
                    'cmd': cmd['cmd']
                }
            }
            if cmd['cmd'] == 'msg':
                obj['text'] = cmd['msg']
                obj['tags']['from'] = cmd['from']
                if 'emotion' in cmd:
                    obj['tags']['emotion'] = cmd['emotion']
            elif cmd['cmd'] == 'scn':
                obj['text'] = cmd['scn']
            obj['id'] = self.get_random_string(32)
            result.append(obj)
        return result

    def await_prompt(self):
        prompt = renpy.exports.input("Enter your reply ('m' for menu)", default="", exclude='{%[]}', length=512)
        prompt = prompt.strip()
        if len(prompt) == 0:
            renpy.exports.jump("eawsw_empty_warning")
            return
        if prompt == "m":
            renpy.exports.jump("eawsw_m_menu")
        else:
            mods = 0
            if 'naomi' in self.mods:
                mods = self.set_bit(mods, 0)
            selected_server = random.choice(self.hosts)
            try:
                renpy.store.m("Waiting for reply...{nw}")
                request_body = json.dumps({
                    'data': self.state['endless_awsw_past'],
                    'execEndpoint': '/',
                    'parameters': {
                        'prompt': prompt
                    }
                })
                self.debug_logger.log("Request: %s" % request_body)
                req = urllib2.Request(
                    '%s/post' % selected_server,
                    headers = {
                        'User-Agent': 'EmeraldOdin/EAWSW',
                        'Content-Type': 'application/json'
                    },
                    data = request_body
                )
                ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
                response = urllib2.urlopen(req, context = ssl_ctx)
                json_str = response.read()
                self.debug_logger.log("Response: %s" % json_str)
                json_response = json.loads(json_str)
                if json_response['header']['status'] is not None:
                    # Error
                    self.last_error = json_response['header']['status']['description']
                    renpy.exports.jump('eawsw_response_error')
                    return
                    
                docs = json_response['data']
                self.execute_commands(docs)
                
                if len(docs) == 0:
                    renpy.exports.jump('eawsw_no_reply_error')
                    return

                if docs[0]['tags']['cmd'] == 'error' and docs[0]['text'] == 'profanity_detected': 
                    renpy.exports.jump('eawsw_profanity_detected')
                    return

                self.state['endless_awsw_past'] += self.eawsw_json_to_doc_array([{
                    'cmd': 'msg',
                    'from': 'c',
                    'msg': prompt
                }])
                self.state['endless_awsw_past'] += docs
                self.strip_past()
            except urllib2.HTTPError as e:
                error_message = e.read()
                with open("eawsw_http_error.log", "w") as f:
                    f.write('Request: %s/post' % selected_server)
                    f.write("\nData: %s" % request_body)
                    f.write("\nError:")
                    f.write(error_message)
                renpy.store.m("HTTP error (stored in eawsw_http_error.log): " + self.sanitize(error_message))
        renpy.exports.block_rollback()
        renpy.exports.jump("eawsw_loop")
    
    def sanitize(self, msg):
        return re.sub(r'[^a-zA-Z0-9_\s"\':\(\)\,\.\-]', '', msg)

    def tick(self):
        if self.state['did_run_start_narrative']:
            self.await_prompt()
        else:
            self.execute_commands(self.state['start_narrative'])
            self.state['endless_awsw_past'] += self.state['start_narrative']
            self.strip_past()
            self.state['did_run_start_narrative'] = True
            renpy.exports.block_rollback()
            renpy.exports.jump("eawsw_loop")

    def get_random_string(self, length):
        charpool = string.ascii_letters + string.digits
        result = ''.join(random.choice(charpool) for i in range(length))
        return result

    def set_bit(self, value, bit):
        return value | (1 << bit)
