init python:
    from eawsw import EAWSWClient, Config
    import json

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

label eawsw_m_menu:
    show zhong normal with dissolve
    $ Zh("Here's the EAWSW v%s menu!" % Config.version)
    menu:
        "New session":
            hide zhong with dissolve
            jump eawsw_pick_your_poison
        "Debug menu":
            jump eawsw_debug_menu
        "Continue":
            hide zhong with dissolve
            jump eawsw_loop

label eawsw_show_past:
    nvl clear
    python:
        result = ""
        for item in eawsw_client.state['endless_awsw_past']:
            result += "%s: %s\n" % (eawsw_client.sanitize(json.dumps(item['tags'])), item['text'])
        n(result)

label eawsw_debug_menu:
    menu:
        "Show past":
            jump eawsw_show_past
        "Back":
            jump eawsw_m_menu

label eawsw_pick_your_poison:
    show zhong normal with dissolve
    Zh "Pick your poison!"
    python:
        def init_client():
            mods = []
            if eawsw_naomi_installed:
                mods.append('naomi')
            hosts = None
            if persistent.eawsw_server is None:
                hosts = Config.public_servers
            else:
                hosts = [persistent.eawsw_server]
            return EAWSWClient(hosts = hosts, mods = mods)
        eawsw_client = init_client()
    menu:
        "You meet Remy at the park.":
            $ eawsw_client.set_start_narrative('park2', [
                { 'cmd': 'msg', 'from': 'c', 'msg': "Hey Remy!" },
                { 'cmd': 'scn', 'scn': 'park2' },
                { 'cmd': 'msg', 'emotion': 'smile', 'from': 'Ry', 'msg': "Hey!" },
            ])
        "You're watching Adine training stunt flights at the beach":
            $ eawsw_client.set_start_narrative('beach', [
                { 'cmd': 'msg', 'from': 'c', 'msg': "Wow nice looping!" },
                { 'cmd': 'scn', 'scn': 'beach' },
                { 'cmd': 'msg', 'emotion': 'giggle', 'from': 'Ad', 'msg': "Thanks! But I have to do much better than this!" },
            ])
        "You're with Lorem in the forest.":
            $ eawsw_client.set_start_narrative('forest1', [
                { 'cmd': 'msg', 'from': 'm', 'msg': "Lorem approached me in the forest." },
                { 'cmd': 'scn', 'scn': 'forest1' },
                { 'cmd': 'msg', 'from': 'Lo', 'emotion': 'happy', 'msg': "Hey!" },
            ])
        "You're with Lorem and Ipsum in their apartment.":
            $ eawsw_client.set_start_narrative('loremapt', [
                { 'cmd': 'scn', 'scn': 'loremapt' },
                { 'cmd': 'msg', 'from': 'Lo', 'emotion': 'happy', 'msg': "I'm glad you came!" },
                { 'cmd': 'scn', 'scn': 'loremapt' },
                { 'cmd': 'msg', 'from': 'Ip', 'emotion': 'happy', 'msg': "I heard all about you." },
            ])
        "You're in a fight with Maverick.":
            $ eawsw_client.set_start_narrative('np1r', [
                { 'cmd': 'scn', 'scn': 'np1r' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "Maverick growled heavily at me." },
                { 'cmd': 'scn', 'scn': 'np1r' },
                { 'cmd': 'msg', 'from': 'Mv', 'emotion': 'angry', 'msg': "I'll slice you open!" },
            ])
        "On a picnic with Bryce":
            $ eawsw_client.set_start_narrative('np2', [
                { 'cmd': 'scn', 'scn': 'np2' },
                { 'cmd': 'msg', 'from': 'm', 'msg': "I sat down with Bryce. During our trip to the picnic place he carried a large basket." },
                { 'cmd': 'scn', 'scn': 'np2' },
                { 'cmd': 'msg', 'from': 'Br', 'emotion': 'laugh', 'msg': "If you're hungry, you can grab something from the fun basket." },
            ])
        "In a shipwreck with Naomi":
            if eawsw_naomi_installed:   
                $ eawsw_client.set_start_narrative('eckoldbiolab', [
                    { 'cmd': 'scn', 'scn': 'eckoldbiolab' },
                    { 'cmd': 'msg', 'from': 'm', 'msg': "After days locked up in here, we still haven't found a way out." },
                    { 'cmd': 'scn', 'scn': 'eckoldbiolab' },
                    { 'cmd': 'msg', 'from': 'Nm', 'emotion': 'blank', 'msg': "Are they looking for us you think?" },
                ])
            else:    
                jump need_naomi_error
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

label eawsw_loop:
    python:
        eawsw_client.tick()
        
        