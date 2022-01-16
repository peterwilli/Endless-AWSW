label pick_your_poison:
    show zhong normal with dissolve
    Zh "Pick your poison!"
    python:
        eawsw_state = {
            'did_run_start_narrative': False,
            'endless_awsw_past': []
        }
    menu:
        "You meet Remy at the park.":
            $ start_narrative = [
                { 'cmd': 'scn', 'scn': 'park2' },
                { 'cmd': 'msg', 'from': 'c', 'msg': "Hey Remy!" },
                { 'cmd': 'scn', 'scn': 'park2' },
                { 'cmd': 'msg', 'from': 'Ry', 'msg': "Hey!" },
            ]
        "You're with Lorem in the forest.":
            $ start_narrative = [
                { 'cmd': 'scn', 'scn': 'forest1' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "Lorem approached me in the forest." },
                { 'cmd': 'scn', 'scn': 'forest1' },
                { 'cmd': 'msg', 'from': 'Lo', 'msg': "Hey!" },
            ]
        "You're with Lorem and Ipsum in their apartment.":
            $ start_narrative = [
                { 'cmd': 'scn', 'scn': 'loremapt' },
                { 'cmd': 'msg', 'from': 'Lo', 'msg': "I'm glad you came!" },
                { 'cmd': 'scn', 'scn': 'loremapt' },
                { 'cmd': 'msg', 'from': 'Ip', 'msg': "I heard all about you." },
            ]
        "You're in a fight with Maverick.":
            $ start_narrative = [
                { 'cmd': 'scn', 'scn': 'np1r' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "Maverick growled heavily at me." },
                { 'cmd': 'scn', 'scn': 'np1r' },
                { 'cmd': 'msg', 'from': 'Mv', 'msg': "I'll slice you open!" },
            ]
        "On a picnic with Bryce":
            $ start_narrative = [
                { 'cmd': 'scn', 'scn': 'np2' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "I sat down with Bryce. During our trip to the picnic place he carried a large basket." },
                { 'cmd': 'scn', 'scn': 'np2' },
                { 'cmd': 'msg', 'from': 'Br', 'msg': "If you're hungry, you can grab something from the fun basket." },
            ]
    jump loop_eawsw

label loop_eawsw:
    python:
        save_past_amount = 6
        class CommandExecutor:
            def __init__(self):
                self.last_scene = None
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
        def await_command():
            import urllib2, urllib
            import json

            prompt = renpy.input(_("Enter your reply"), default="", exclude='{%[]}', length=512)
            prompt = prompt.strip()
            
            if prompt == "clear":
                renpy.jump("pick_your_poison")
            else:
                query = urllib.urlencode({
                    'past': json.dumps(eawsw_state['endless_awsw_past']),
                    'prompt': prompt
                })
                req = urllib2.Request('http://127.0.0.1:5000/get_command?%s' % query)
                response = urllib2.urlopen(req)
                json_str = response.read()
                command_dict = json.loads(json_str)
                cmds = command_dict['cmds']
                command_executor.execute_commands(cmds)

                eawsw_state['endless_awsw_past'] += cmds
                eawsw_state['endless_awsw_past'] = eawsw_state['endless_awsw_past'][save_past_amount * -1:]
            renpy.block_rollback()
            renpy.jump("loop_eawsw")
        if eawsw_state['did_run_start_narrative']:
            await_command()
        else:
            command_executor.execute_commands(start_narrative)
            eawsw_state['endless_awsw_past'] += start_narrative
            eawsw_state['endless_awsw_past'] = eawsw_state['endless_awsw_past'][save_past_amount * -1:]
            eawsw_state['did_run_start_narrative'] = True
            renpy.block_rollback()
            renpy.jump("loop_eawsw")