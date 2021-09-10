label me_testmod_sebmeet:
    $ prompt = renpy.input(_("Enter your reply"), default="", exclude='{%,[,]}', length=512)

    python:
        import urllib2, urllib
        import json
        if persistent.endless_awsw_past is None:
            persistent.endless_awsw_past = ""
        query = urllib.urlencode({
            'past': persistent.endless_awsw_past,
            'prompt': prompt
        })
        req = urllib2.Request('http://127.0.0.1:5000/get_command?%s' % query)
        response = urllib2.urlopen(req)
        json_str = response.read()
        command_dict = json.loads(json_str)
        cmd = command_dict['cmd']
        
    if cmd == "msg":
        $ msg_from = command_dict['from']
        $ msg = command_dict['msg']
        $ persistent.endless_awsw_past += msg
        if msg_from == "Rz":
            Rz "[msg]"