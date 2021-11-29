import renpy
import os
import json
import re
import glob

def extract_for_training(nodes, last_scene = None):
    if nodes is None:
        return None
    result = []
    state = {
        'last_speaker': None,
        'buffer': []
    }
    action_menu_regex = re.compile(r'\[\[.*?\]')
    character_images_regex = re.compile(r'{#.*?}(.*)')
    regular_images_regex = re.compile(r'{image=.*?}')
    
    def safe_buffer_append(line):
        line = line.strip()
        if len(line) > 0:
            state['buffer'].append(line)

    def safe_result_append(line):
        line = line.strip()
        if len(line) > 0:
            result.append(line)
        state['last_speaker'] = None

    def process_say(node):
        allowed_lines = ["n", "m", "Rz", "Lo", "Ad", "c", "Ry", "Mv", "Br", "An", "Ip", "Sb", "Wr", "Zh", "Kv", "Ka", "Em"]
        info = node.diff_info()
        if info[1] in allowed_lines:
            commands = []
            if info[1] != state['last_speaker']:
                if info[1] == "c":
                    prefix = "PlayerReply"
                else:
                    prefix = "DragonReply"
                commands.append(prefix)
            commands.append(info[1])                
            should_end_buffer = False
            if info[1] == "c" and state['last_speaker'] != None and state['last_speaker'] != "c":
                should_end_buffer = True
            if should_end_buffer:
                safe_result_append(" ".join(state['buffer']))
                state['buffer'] = []
            commands.append("\"{}\"".format(info[2]))
            if last_scene is not None:
                commands.append("scn {}".format(last_scene))
            safe_buffer_append(" ".join(commands))
        state['last_speaker'] = info[1]

    for node in nodes:
        if isinstance(node, renpy.ast.Menu):
            if len(state['buffer']) > 0:
                pre_menu = " ".join(state['buffer'])
            else:
                pre_menu = None
            state['buffer'] = []
            for menu_item in node.items:
                menu_str = menu_item[0].strip()
                forbidden_menu_items = [
                    "Yes. I want to skip ahead.",
                    "No. Don't skip ahead."
                ]
                if menu_str in forbidden_menu_items:
                    break
                menu_str = re.sub(character_images_regex, r"\1", menu_str)
                menu_str = re.sub(regular_images_regex, r"", menu_str)
                player_prefix = "c"
                if state['last_speaker'] != "c":
                    player_prefix = "PlayerReply " + player_prefix
                if pre_menu is None:
                    safe_buffer_append("{} \"{}\"".format(player_prefix, menu_str))
                else:
                    safe_buffer_append("{} {} \"{}\"".format(pre_menu, player_prefix, menu_str))
                safe_buffer_append(extract_for_training(menu_item[2]))
                safe_result_append(" ".join(state['buffer']))
                state['buffer'] = []
                state['last_speaker'] = None
        elif isinstance(node, renpy.ast.Say):
            process_say(node)
        elif isinstance(node, renpy.ast.Scene):
            last_scene = node.diff_info()[1][0]
        elif isinstance(node, renpy.ast.If):
            filtered_nodes = []
            for entry in node.entries:
                for inner_node in entry[1]:
                    if isinstance(inner_node, renpy.ast.Say):
                        filtered_nodes.append(inner_node)
            if len(filtered_nodes) > 1:
                # More than 1 say node means we probably have a branch
                for entry in node.entries:
                    extracted_if_result = extract_for_training(entry[1])
                    safe_result_append(extracted_if_result)
            elif len(filtered_nodes) > 0:
                process_say(filtered_nodes[0])
        
    # add any excess replies to the last line...
    excess = ""
    if len(result) > 0:
        excess += " "
    excess += " ".join(state['buffer'])
    return "\n".join(result) + excess

def parse():
    script_folder = os.path.dirname(os.path.realpath(__file__))
    awsw_path = os.path.join(script_folder, "..", "Angels with Scaly Wings", "game")
    # awsw_path = os.path.join(script_folder, "test_rpy")
    rpy_files = glob.glob(os.path.join(awsw_path, "*.rpy"))
    with open("training_data.txt", 'w') as training_data_fd:
        for rpy_file in rpy_files:
            print("Parsing %s" % rpy_file)
            ast = renpy.parser.parse(rpy_file)
            result = extract_for_training(ast)
            if result is not None:
                training_data_fd.write(result)