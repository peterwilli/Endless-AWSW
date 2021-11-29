import renpy
import os
import json
import re
import glob

def extract_for_training(nodes):
    if nodes is None:
        return None
    result = []
    buffer = []
    last_scene = None
    last_speaker = None
    action_menu_regex = re.compile(r'\[\[.*?\]')
    character_images_regex = re.compile(r'{#.*?}(.*)')
    regular_images_regex = re.compile(r'{image=.*?}')
    
    def safe_buffer_append(line):
        line = line.strip()
        if len(line) > 0:
            buffer.append(line)

    def safe_result_append(line):
        line = line.strip()
        if len(line) > 0:
            result.append(line)
        last_speaker = None

    for node in nodes:
        if isinstance(node, renpy.ast.Menu):
            if len(buffer) > 0:
                pre_menu = " ".join(buffer)
            else:
                pre_menu = None
            buffer = []
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
                if last_speaker != "c":
                    player_prefix = "PlayerReply " + player_prefix
                if pre_menu is None:
                    safe_buffer_append("{} \"{}\"".format(player_prefix, menu_str))
                else:
                    safe_buffer_append("{} {} \"{}\"".format(pre_menu, player_prefix, menu_str))
                safe_buffer_append(extract_for_training(menu_item[2]))
                safe_result_append(" ".join(buffer))
                buffer = []
                last_speaker = None
        elif isinstance(node, renpy.ast.Say):
            allowed_lines = ["n", "m", "Rz", "Lo", "Ad", "c", "Ry", "Mv", "Br", "An", "Ip", "Sb", "Wr", "Zh", "Kv", "Ka", "Em"]
            info = node.diff_info()
            if info[1] in allowed_lines:
                commands = []
                if info[1] != last_speaker:
                    if info[1] == "c":
                        prefix = "PlayerReply"
                    else:
                        prefix = "DragonReply"
                    commands.append(prefix)
                commands.append(info[1])                
                should_end_buffer = False
                if info[1] == "c" and last_speaker != None and last_speaker != "c":
                    should_end_buffer = True
                if should_end_buffer:
                    safe_result_append(" ".join(buffer))
                    buffer = []
                commands.append("\"{}\"".format(info[2]))
                if last_scene is not None:
                    commands.append("scn {}".format(last_scene))
                safe_buffer_append(" ".join(commands))
            last_speaker = info[1]
        elif isinstance(node, renpy.ast.Scene):
            last_scene = node.diff_info()[1][0]
        elif isinstance(node, renpy.ast.If):
            for entry in node.entries:
                extracted_if_result = extract_for_training(entry[1])
                safe_result_append(extracted_if_result)
        
    # add any excess replies to the last line...
    excess = ""
    if len(result) > 0:
        excess += " "
    excess += " ".join(buffer)
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