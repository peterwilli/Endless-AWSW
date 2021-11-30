label pick_your_poison:
    show zhong normal with dissolve
    Zh "Pick your poison!"
    menu:
        "You meet Remy at the park.":
            $ persistent.endless_awsw_past = ['PlayerReply c "Hey Remy!"', 'DragonReply Ry "Hey!"']
            scene town2 at Pan ((300, 400), (150, 400), 3.0) with dissolveslow
            show remy normal with dissolve
            Ry smile "Hey!"
        "You're with Lorem in the forest.":
            $ persistent.endless_awsw_past = ['DragonReply m "Lorem approached me in the forest."', 'DragonReply Lo "Hey!"']
            scene forest1 at Pan ((0, 360), (0,0), 8.0) with dissolveslow
            show lorem happy with dissolve
            Lo happy "Hey!"
        #"You're in a fight with Maverick.":
            # # persistent.endless_awsw_past = 'DragonReply m "Maverick growled heavily at me." DragonReply M "Hey!"'
            # scene forest1 at Pan ((0, 360), (0,0), 8.0) with dissolveslow
    jump loop_eawsw

label loop_eawsw:
    python:
        # Putting this in a function so RenPy doesn't save it.
        def get_character_mapping():
            return {
                'Ry': 'remy',
                'Lo': 'lorem',
                'Br': 'bryce',
                'Wr': None,
                'Ka': 'katsu',
                'Rz': 'reza',
                'Kv': 'kevin',
                'Mv': 'maverick',
                'Zh': 'zhong',
                'm': None,
                'An': 'anna',
                'Ad': 'adine',
                'Sb': 'sebastian',
                'Ip': 'ipsum'
            }

        def hide_all_characters():
            character_mapping = get_character_mapping()
            for character in character_mapping.values():
                if character is not None:
                    renpy.hide(character)

        def await_command():
            import urllib2, urllib
            import json

            talk_functions = {
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
                'm': m,
                'An': An,
                'Ad': Ad,
                'Sb': Sb
            }

            prompt = renpy.input(_("Enter your reply"), default="", exclude='{%,[,]}', length=512)
            prompt = prompt.strip()
            
            if prompt == "clear":
                renpy.jump("pick_your_poison")
            else:
                query = urllib.urlencode({
                    'past': " ".join(persistent.endless_awsw_past),
                    'prompt': prompt
                })
                req = urllib2.Request('http://127.0.0.1:5000/get_command?%s' % query)
                response = urllib2.urlopen(req)
                json_str = response.read()
                command_dict = json.loads(json_str)
                cmd = command_dict['cmd']

            if cmd == "msg":
                character_mapping = get_character_mapping()
                hide_all_characters()
                msg_from = command_dict['from']
                msg = command_dict['msg']
                persistent.endless_awsw_past += [msg]
                renpy.show('%s normal b' % (character_mapping[msg_from]))
                # Only save the last 3
                persistent.endless_awsw_past = persistent.endless_awsw_past[-3:]
                talk_functions[msg_from](msg)
                # Todo custom expressions
                renpy.jump("loop_eawsw")
        await_command()