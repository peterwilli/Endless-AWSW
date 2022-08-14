init python:
    import random
    import string
    def eawsw_get_random_string(length):
        charpool = string.ascii_letters + string.digits
        result = ''.join(random.choice(charpool) for i in range(length))
        return result

    # Since we can't use Jina's DocumentArray directly in this mod
    # We use a sexy json and convert it to DocumentArray's json format
    def eawsw_json_to_doc_array(arr):
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
            obj['id'] = eawsw_get_random_string(32)
            result.append(obj)
        return result

label eawsw_intro:
    show zhong normal with dissolve
    Zh "Hello [player_name]! I'm your Endless Angels with Scaly Wings host for today. Do you wish to proceed?"
    menu:
        "I wish to play endless":
            jump eawsw_server_selection
        "I wish to play normally":
            hide zhong with dissolve
            jump eawsw_intro_return

label eawsw_server_selection:
    if persistent.eawsw_server is not None:
        Zh shy b "It seems you already selected a custom server before. Do you wish to use your last selected server?"
        menu:
            "Use '[persistent.eawsw_server]'":
                jump eawsw_pick_your_poison
            "Choose new server":
                $ renpy.pause (0.5)
    show zhong smile with dissolve
    Zh "That's great! Do you wish to use one of our public servers or host your own? The servers are used for hosting the AI-model, no data except your prompts are sent. If you use your own server, no data is sent, but you need to have a beefy computer and some computer skills to run it."
    menu:
        "Use a public / free server":
            $ persistent.eawsw_server = None
        "Use my own":
            python:
                server_input = renpy.input(_("Type your server URL"), default="http://localhost:5000", exclude='{%[]}', length=512)
                server_input = server_input.strip()
                persistent.eawsw_server = server_input
    jump eawsw_pick_your_poison

label eawsw_pick_your_poison:
    show zhong normal with dissolve
    Zh "Pick your poison!"
    python:
        eawsw_state = {
            'did_run_start_narrative': False,
            'endless_awsw_past': [],
            'start_scene': None
        }      
    menu:
        "You meet Remy at the park.":
            $ eawsw_state['start_scene'] = 'park2'
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'msg', 'from': 'c', 'msg': "Hey Remy!" },
                { 'cmd': 'scn', 'scn': 'park2' },
                { 'cmd': 'msg', 'emotion': 'smile', 'from': 'Ry', 'msg': "Hey!" },
            ]
        "You're watching Adine training stunt flights at the beach":
            $ eawsw_state['start_scene'] = 'beach'
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'msg', 'from': 'c', 'msg': "Wow nice looping!" },
                { 'cmd': 'scn', 'scn': 'beach' },
                { 'cmd': 'msg', 'emotion': 'giggle', 'from': 'Ad', 'msg': "Thanks! But I have to do much better than this!" },
            ]
        "You're with Lorem in the forest.":
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'scn', 'scn': 'forest1' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "Lorem approached me in the forest." },
                { 'cmd': 'scn', 'scn': 'forest1' },
                { 'cmd': 'msg', 'from': 'Lo', 'emotion': 'happy', 'msg': "Hey!" },
            ]
        "You're with Lorem and Ipsum in their apartment.":
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'scn', 'scn': 'loremapt' },
                { 'cmd': 'msg', 'from': 'Lo', 'emotion': 'happy', 'msg': "I'm glad you came!" },
                { 'cmd': 'scn', 'scn': 'loremapt' },
                { 'cmd': 'msg', 'from': 'Ip', 'emotion': 'happy', 'msg': "I heard all about you." },
            ]
        "You're in a fight with Maverick.":
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'scn', 'scn': 'np1r' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "Maverick growled heavily at me." },
                { 'cmd': 'scn', 'scn': 'np1r' },
                { 'cmd': 'msg', 'from': 'Mv', 'emotion': 'angry', 'msg': "I'll slice you open!" },
            ]
        "On a picnic with Bryce":
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'scn', 'scn': 'np2' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "I sat down with Bryce. During our trip to the picnic place he carried a large basket." },
                { 'cmd': 'scn', 'scn': 'np2' },
                { 'cmd': 'msg', 'from': 'Br', 'emotion': 'laugh', 'msg': "If you're hungry, you can grab something from the fun basket." },
            ]
        "In a shipwreck with Naomi":
            if eawsw_naomi_installed:   
                $ eawsw_state['start_narrative'] = [
                    { 'cmd': 'scn', 'scn': 'eckoldbiolab' },
                    { 'cmd': 'msg', 'from': 'm', 'msg': "After days locked up in here, we still haven't found a way out." },
                    { 'cmd': 'scn', 'scn': 'eckoldbiolab' },
                    { 'cmd': 'msg', 'from': 'Nm', 'emotion': 'blank', 'msg': "Are they looking for us you think?" },
                ]
            else:    
                jump need_naomi_error
    $ eawsw_state['start_narrative'] = eawsw_json_to_doc_array(eawsw_state['start_narrative'])
    jump eawsw_loop

label eawsw_empty_warning:
    show maverick angry with dissolve
    Mv "How are dragons supposed to reply to an empty message?!"
    hide maverick with dissolve
    pause(0.5)
    jump eawsw_loop

label eawsw_profanity_detected:
    show maverick angry with dissolve
    Mv "I detected profanity or NFSW langague in your prompt. This is not allowed on the public server! You can run your own from source if you wish to do this, or change your prompt!"
    hide maverick with dissolve
    pause(0.5)
    jump eawsw_loop

label need_naomi_error:
    show maverick nice with dissolve
    Mv "You need the mod 'A Solitary Mind' to play with Naomi! Go get it from the workshop and restart the game, otherwise select another narrative!"
    hide maverick with dissolve
    pause(0.5)
    jump eawsw_pick_your_poison

label eawsw_no_reply_error:
    show maverick nice with dissolve
    Mv "Sorry, the AI couldn't form a reply after multiple tries, please try another prompt!"
    hide maverick with dissolve
    pause(0.5)
    jump eawsw_loop

label eawsw_version:
    show maverick nice with dissolve
    Mv "You're using EAWSW v0.01"
    hide maverick with dissolve
    pause(0.5)
    jump eawsw_loop

label eawsw_loop:
    python:
        import re
        import urllib2, urllib
        import json
        import ssl
        import random

        # Jina is currently providing EAWSW hosting for free on their JCloud service.
        # At the moment, there's no need for community services.
        public_servers = ['https://e8495ef0a2.wolf.jina.ai']
        save_past_amount = 6

        def set_bit(value, bit):
            return value | (1 << bit)

        def clear_bit(value, bit):
            return value & ~(1 << bit)

        class DebugLogger:
            def __init__(self):
                self.debug_mode = True
            
            def log(self, msg):
                if self.debug_mode:
                    with open("eawsw_logs.log", "a") as f:
                        f.write(msg + "\n")

        class CommandExecutor:
            def __init__(self):
                self.last_scene = eawsw_state['start_scene']
                self.last_character = None
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
                    "Ad": Ad,
                    "An": An,
                    "Br": Br,
                    "Dm": Dm,
                    "Em": Em,
                    "Ip": Ip,
                    "Iz": Iz,
                    "Ka": Ka,
                    "Kv": Kv,
                    "Lo": Lo,
                    "Mv": Mv,
                    "Ry": Ry,
                    "Rz": Rz,
                    "Sb": Sb,
                    "Zh": Zh
                }
                if eawsw_naomi_installed:
                    self.talk_functions['Nm'] = Nm
                # Add self
                self.talk_functions.update({
                    'm': m,
                    'c': c
                })

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
                        renpy.scene()
                        if self.last_scene is not None:
                            renpy.show(self.last_scene)
                        if self.last_character is not None:
                            renpy.show(self.last_character)
                        if msg_from in self.talk_functions:
                            talk_fn = self.talk_functions[msg_from]
                            talk_fn(msg)
                        
        command_executor = CommandExecutor()
        debug_logger = DebugLogger()

        def strip_past():
            eawsw_state['endless_awsw_past'] = eawsw_state['endless_awsw_past'][save_past_amount * -1:]
            potential_stray_dragon = eawsw_state['endless_awsw_past'][0]
            if potential_stray_dragon['tags']['cmd'] == 'msg' and potential_stray_dragon['tags']['from'] != 'c':
                # Dragon without a scene, forbidden so we delete it too
                eawsw_state['endless_awsw_past'].pop(0)
            
        def sanitize(msg):
            return re.sub(r'[^a-zA-Z0-9_\s]', '', msg)

        def await_command():
            prompt = renpy.input(_("Enter your reply"), default="", exclude='{%[]}', length=512)
            prompt = prompt.strip()
            if len(prompt) == 0:
                renpy.jump("eawsw_empty_warning")
                return
            
            if prompt == "clear":
                renpy.jump("eawsw_pick_your_poison")
            elif prompt == "version":
                renpy.jump("eawsw_version")
            else:
                mods = 0
                if eawsw_naomi_installed:
                    mods = set_bit(mods, 0)
                selected_server = None
                if persistent.eawsw_server is None:
                    # Use a public server from a list
                    selected_server = random.choice(public_servers)
                else:
                    selected_server = persistent.eawsw_server
                try:
                    m("Waiting for reply...{nw}")
                    request_body = json.dumps({
                        'data': eawsw_state['endless_awsw_past'],
                        'execEndpoint': '/',
                        'parameters': {
                            'prompt': prompt
                        }
                    })
                    debug_logger.log("Request: %s" % request_body)
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
                    debug_logger.log("Response: %s" % json_str)
                    docs = json.loads(json_str)['data']
                    command_executor.execute_commands(docs)
                    
                    if len(docs) == 0:
                        renpy.jump('eawsw_no_reply_error')
                        return

                    if docs[0]['tags']['cmd'] == 'error' and docs[0]['text'] == 'profanity_detected': 
                        renpy.jump('eawsw_profanity_detected')
                        return

                    eawsw_state['endless_awsw_past'] += eawsw_json_to_doc_array([{
                        'cmd': 'msg',
                        'from': 'c',
                        'msg': prompt
                    }])
                    eawsw_state['endless_awsw_past'] += docs
                    strip_past()
                except urllib2.HTTPError as e:
                    error_message = e.read()
                    with open("eawsw_http_error.log", "w") as f:
                        f.write('Request: %s/post' % selected_server)
                        f.write("\nData: %s" % request_body)
                        f.write("\nError:")
                        f.write(error_message)
                    m("HTTP error (stored in eawsw_http_error.log): " + sanitize(error_message))
            renpy.block_rollback()
            renpy.jump("eawsw_loop")
        if eawsw_state['did_run_start_narrative']:
            await_command()
        else:
            command_executor.execute_commands(eawsw_state['start_narrative'])
            eawsw_state['endless_awsw_past'] += eawsw_state['start_narrative']
            strip_past()
            eawsw_state['did_run_start_narrative'] = True
            renpy.block_rollback()
            renpy.jump("eawsw_loop")