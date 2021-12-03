label pick_your_poison:
    show zhong normal with dissolve
    Zh "Pick your poison!"
    menu:
        "You meet Remy at the park.":
            $ persistent.endless_awsw_past = ['<p><msg>c "Hey Remy!"', '<d><scn>park2<msg>Ry "Hey!"']
            scene town2 at Pan ((300, 400), (150, 400), 3.0) with dissolveslow
            show remy normal with dissolve
            Ry smile "Hey!"
        "You're with Lorem in the forest.":
            $ persistent.endless_awsw_past = ['<d><scn>forest1<msg>m "Lorem approached me in the forest."', '<d><scn>forest1<msg>Lo "Hey!"']
            scene forest1 at Pan ((0, 360), (0,0), 8.0) with dissolveslow
            show lorem happy with dissolve
            Lo happy "Hey!"
        #"You're in a fight with Maverick.":
            # # persistent.endless_awsw_past = 'DragonReply m "Maverick growled heavily at me." DragonReply M "Hey!"'
            # scene forest1 at Pan ((0, 360), (0,0), 8.0) with dissolveslow
    jump loop_eawsw

label loop_eawsw:
    python:
        last_scene = None

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
                'Em': 'emera',
                'm': None,
                'An': 'anna',
                'Ad': 'adine',
                'Sb': 'sebastian',
                'Ip': 'ipsum'
            }

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
                'Em': Em,
                'm': m,
                'c': c,
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
                cmds = command_dict['cmds']
            for item in cmds:
                cmd = item['cmd']
                if cmd == "scn":
                    last_scene = item['scn']
                elif cmd == "msg":
                    character_mapping = get_character_mapping()
                    if last_scene is not None:
                        renpy.scene()   
                        renpy.show(last_scene)
                    msg_from = item['from']
                    msg = item['msg']
                    persistent.endless_awsw_past += [msg]
                    # Todo custom expressions
                    renpy.show('%s normal b' % (character_mapping[msg_from]))
                    talk_functions[msg_from](msg)
            # Only save the last 3
            persistent.endless_awsw_past = persistent.endless_awsw_past[-3:]
            renpy.jump("loop_eawsw")
        await_command()