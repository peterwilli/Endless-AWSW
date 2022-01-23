label eawsw_intro:
    show zhong normal with dissolve
    Zh "Hello [player_name]! I'm your Endless Angels with Scaly Wings host for today. Do you wish to proceed?"
    menu:
        "I wish to play endless":
            jump eawsw_server_selection
        "I wish to play normally":
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
                { 'cmd': 'msg', 'from': 'Ry', 'msg': "Hey!" },
            ]
        "You're with Lorem in the forest.":
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'scn', 'scn': 'forest1' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "Lorem approached me in the forest." },
                { 'cmd': 'scn', 'scn': 'forest1' },
                { 'cmd': 'msg', 'from': 'Lo', 'msg': "Hey!" },
            ]
        "You're with Lorem and Ipsum in their apartment.":
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'scn', 'scn': 'loremapt' },
                { 'cmd': 'msg', 'from': 'Lo', 'msg': "I'm glad you came!" },
                { 'cmd': 'scn', 'scn': 'loremapt' },
                { 'cmd': 'msg', 'from': 'Ip', 'msg': "I heard all about you." },
            ]
        "You're in a fight with Maverick.":
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'scn', 'scn': 'np1r' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "Maverick growled heavily at me." },
                { 'cmd': 'scn', 'scn': 'np1r' },
                { 'cmd': 'msg', 'from': 'Mv', 'msg': "I'll slice you open!" },
            ]
        "On a picnic with Bryce":
            $ eawsw_state['start_narrative'] = [
                { 'cmd': 'scn', 'scn': 'np2' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "I sat down with Bryce. During our trip to the picnic place he carried a large basket." },
                { 'cmd': 'scn', 'scn': 'np2' },
                { 'cmd': 'msg', 'from': 'Br', 'msg': "If you're hungry, you can grab something from the fun basket." },
            ]
    jump eawsw_loop

label eawsw_empty_warning:
    show maverick angry with dissolve
    Mv "How are dragons supposed to reply to an empty message?!"
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
        # If you maintain a public server, feel free to add it.
        public_servers = ['https://eawsw_api.emeraldodin.com']
        save_past_amount = 12
        class CommandExecutor:
            def __init__(self):
                self.last_scene = eawsw_state['start_scene']
                self.last_character = None
                self.character_mapping = {
                    'Ry': 'remy',
                    'Lo': 'lorem',
                    'Br': 'bryce',
                    'Wr': None,
                    'Ka': 'katsu',
                    'Rz': 'reza',
                    'Kv': 'kevin',
                    'Mv': 'maverick',
                    'Zh': 'zhong',
                    'Em': 'emera',
                    'm': None,
                    'An': 'anna',
                    'Ad': 'adine',
                    'Sb': 'sebastian',
                    'Ip': 'ipsum'
                }

                self.talk_functions = {
                    'Ry': Ry,
                    'Lo': Lo,
                    'Ip': Ip,
                    'Br': Br,
                    'Wr': Wr,
                    'Ka': Ka,
                    'Rz': Rz,
                    'Kv': Kv,
                    'Zh': Zh,
                    'Mv': Mv,
                    'Em': Em,
                    'm': m,
                    'c': c,
                    'An': An,
                    'Ad': Ad,
                    'Sb': Sb
                }

            def execute_commands(self, cmds):
                for item in cmds:
                    cmd = item['cmd']
                    if cmd == "scn":
                        self.last_scene = item['scn']
                    elif cmd == "msg":
                        msg_from = item['from']
                        msg = item['msg']
                        # Todo custom expressions
                        if msg_from in self.character_mapping:
                            if self.character_mapping[msg_from] is None:
                                self.last_character = None
                            else:
                                self.last_character = '%s normal b' % (self.character_mapping[msg_from])
                        renpy.scene()
                        if self.last_scene is not None:
                            renpy.show(self.last_scene)
                        if self.last_character is not None:
                            renpy.show(self.last_character)
                        if msg_from in self.talk_functions:
                            talk_fn = self.talk_functions[msg_from]
                            talk_fn(msg)
                        
        command_executor = CommandExecutor()
        def strip_past():
            eawsw_state['endless_awsw_past'] = eawsw_state['endless_awsw_past'][save_past_amount * -1:]
            potential_stray_dragon = eawsw_state['endless_awsw_past'][0]
            if potential_stray_dragon['cmd'] == 'msg' and potential_stray_dragon['from'] != 'c':
                # Dragon without a scene, forbidden so we delete it too
                eawsw_state['endless_awsw_past'].pop(0)

        def await_command():
            import urllib2, urllib
            import json
            import random

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
                query = urllib.urlencode({
                    'past': json.dumps(eawsw_state['endless_awsw_past']),
                    'prompt': prompt
                })
                selected_server = None
                if persistent.eawsw_server is None:
                    # Use a public server from a list
                    selected_server = random.choice(public_servers)
                else:
                    selected_server = persistent.eawsw_server
                try:
                    req = urllib2.Request(
                        '%s/get_command?%s' % (selected_server, query),
                        headers = {
                            'User-Agent': 'EmeraldOdin/EAWSW'
                        }
                    )
                    response = urllib2.urlopen(req)
                    json_str = response.read()
                    command_dict = json.loads(json_str)
                    cmds = command_dict['cmds']
                    command_executor.execute_commands(cmds)
                    if len(cmds) > 0:
                        eawsw_state['endless_awsw_past'] += [{
                            'cmd': 'msg',
                            'from': 'c',
                            'msg': prompt
                        }]
                    eawsw_state['endless_awsw_past'] += cmds
                    strip_past()
                except urllib2.HTTPError as e:
                    error_message = e.read()
                    m("HTTP error: " + error_message)
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