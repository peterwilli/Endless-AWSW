import renpy
import os
import json
import re
import glob

def extract_sentiment(nodes, state = None):
    if nodes is None:
        return None
        
    if state is None:
        state = {
            'sentiment_pairs': []
        }

    def process_say(node):
        allowed_lines = ["n", "m", "Rz", "Lo", "Ad", "c", "Ry", "Mv", "Br", "An", "Ip", "Sb", "Wr", "Zh", "Kv", "Ka", "Em"]
        info = node.diff_info()
        if info[1] in allowed_lines:
            sentiment = None
            skip_attrs = ['flip']
            if node.attributes is not None:
                for attr in node.attributes:
                    if not attr in skip_attrs:
                        sentiment = attr
                        break
                if sentiment is not None:
                    state['sentiment_pairs'].append({
                        'text': info[2],
                        'dragon': info[1],
                        'sentiment': sentiment 
                    })
    for node in nodes:
        if isinstance(node, renpy.ast.Say):
            process_say(node)
        elif isinstance(node, renpy.ast.If):
            for entry in node.entries:
                extract_sentiment(entry[1], state)
        elif isinstance(node, renpy.ast.Label):
            extract_sentiment(node.block, state)
        elif isinstance(node, renpy.ast.Menu):
            for menu_item in node.items:
                extract_sentiment(menu_item[2], state)
    return "\n".join(["%s %s %s" % (sp['dragon'], sp['sentiment'], sp['text']) for sp in state['sentiment_pairs']])
    
def extract_for_training(nodes, state = None):
    if nodes is None:
        return None
    result = []
    if state is None:
        state = {
            'last_speaker': None,
            'buffer': [],
            'last_scene': None
        }
    action_menu_regex = re.compile(r'\[\[.*?\]')
    character_images_regex = re.compile(r'{#.*?}(.*)')
    regular_images_regex = re.compile(r'{image=.*?}')
    
    def safe_buffer_append(line):
        if line is not None:
            line = line.strip()
            if len(line) > 0:
                state['buffer'].append(line)

    def safe_result_append(line):
        line = line.strip()
        if len(line) > 0:
            result.append(line)
        state['last_speaker'] = None

    def clean_say_line(line):
        line = line.strip()
        split = line.split("\n")
        result = []
        for part in split:
            result.append(part.strip())
        return "\\n".join(result)

    def process_say(node):
        allowed_lines = ["n", "m", "Rz", "Lo", "Ad", "c", "Ry", "Mv", "Br", "An", "Ip", "Sb", "Wr", "Zh", "Kv", "Ka", "Em"]
        info = node.diff_info()
        if info[1] in allowed_lines:
            commands = []
            if True or info[1] != state['last_speaker']:
                if info[1] == "c":
                    prefix = "<p>"
                else:
                    prefix = "<d>"
                commands.append(prefix)
            if state['last_scene'] is not None and info[1] != "c":
                commands.append("<scn>{}".format(state['last_scene']))
                #state['last_scene'] = None
            commands.append('<msg>')                
            commands.append(info[1] + " ")                
            should_end_buffer = False
            if info[1] == "c" and state['last_speaker'] != None and state['last_speaker'] != "c":
                should_end_buffer = True
            if should_end_buffer:
                safe_result_append("".join(state['buffer']))
                state['buffer'] = []
            commands.append("\"{}\"".format(clean_say_line(info[2])))
            safe_buffer_append("".join(commands))
        state['last_speaker'] = info[1]

    for node in nodes:
        if isinstance(node, renpy.ast.Menu):
            if len(state['buffer']) > 0:
                pre_menu = "".join(state['buffer'])
            else:
                pre_menu = None
            state['buffer'] = []
            forbidden_menu_items = [
                "Yes. I want to skip ahead.",
                "No. Don't skip ahead."
            ]
            for menu_item in node.items:
                menu_str = menu_item[0].strip()
                if menu_str in forbidden_menu_items:
                    break
                is_ok = False
                if menu_item[2] is None:
                    break
                for node2 in menu_item[2]:
                    if isinstance(node2, renpy.ast.Say):
                        is_ok = True
                        break
                if not is_ok:
                    break
                menu_str = re.sub(character_images_regex, r"\1", menu_str)
                menu_str = re.sub(regular_images_regex, r"", menu_str)
                player_prefix = "c"
                if state['last_speaker'] != player_prefix:
                    player_prefix = "<p><msg>" + player_prefix
                    state['last_speaker'] = player_prefix
                if pre_menu is None:
                    safe_buffer_append("{} \"{}\"".format(player_prefix, menu_str))
                else:
                    safe_buffer_append("{}{} \"{}\"".format(pre_menu, player_prefix, menu_str))
                nested_state = dict(state)
                nested_state['buffer'] = []
                safe_buffer_append(extract_for_training(menu_item[2], nested_state))
                safe_result_append("".join(state['buffer']))
                state['buffer'] = []
                # state['last_speaker'] = None
        elif isinstance(node, renpy.ast.Say):
            process_say(node)
        elif isinstance(node, renpy.ast.Label):
            extracted_label = extract_for_training(node.block, state)
            safe_result_append(extracted_label)
        elif isinstance(node, renpy.ast.Scene):
            state['last_scene'] = node.diff_info()[1][0]
        elif isinstance(node, renpy.ast.If):
            filtered_nodes = []
            for entry in node.entries:
                for inner_node in entry[1]:
                    if isinstance(inner_node, renpy.ast.Say):
                        filtered_nodes.append(inner_node)
            if len(filtered_nodes) > 1:
                # More than 1 say node means we probably have a branch
                for entry in node.entries:
                    extracted_if_result = extract_for_training(entry[1], state)
                    safe_result_append(extracted_if_result)
            elif len(filtered_nodes) > 0:
                process_say(filtered_nodes[0])
    # add any excess replies to the last line...
    excess = "".join(state['buffer'])
    return "\n".join(result) + excess

def parse():
    script_folder = os.path.dirname(os.path.realpath(__file__))
    awsw_path = os.path.join(script_folder, "..", "Angels with Scaly Wings", "game")
    #awsw_path = os.path.join(script_folder, "test_rpy")
    rpy_files = glob.glob(os.path.join(awsw_path, "*.rpy"))
    with open("training_data.txt", 'w') as training_data_fd:
        with open("sentiment_training_data.txt", 'w') as sentiment_data_fd:
            blacklist = ["screens.rpy", "status.rpy", "help.rpy", "achievements.rpy", "gallery.rpy"]
            for rpy_file in rpy_files:
                file_name = os.path.basename(rpy_file)
                if not file_name in blacklist:
                    ast = renpy.parser.parse(rpy_file)
                    print("Parsing %s" % rpy_file)
                    if ast is None:
                        print("%s has empty AST..." % rpy_file)
                    else:
                        result_data = extract_for_training(ast)
                        if result_data is not None:
                            training_data_fd.write(result_data)
                        result_sentiment = extract_sentiment(ast)
                        if result_sentiment is not None:
                            sentiment_data_fd.write(result_sentiment + "\n")