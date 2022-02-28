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
    action_menu_regex = re.compile(r'\[\[.*?\]')
    character_images_regex = re.compile(r'{#.*?}(.*)')
    regular_images_regex = re.compile(r'{image=.*?}')
    
    if state is None:
        state = {
            'lines': [],
            'current_line': '',
            'last_speaker': None,
            'last_scene': None
        }

    def clean_say_line(line):
        line = line.strip()
        split = line.split("\n")
        result = []
        for part in split:
            result.append(part.strip())
        return "\\n".join(result)

    def end_buffer(state):
        if len(state['current_line']) > 0:
            state['lines'].append(state['current_line'])
        state['current_line'] = ''
        state['last_speaker'] = None

    def process_say(character, msg):
        allowed_lines = ["n", "m", "Rz", "Lo", "Ad", "c", "Ry", "Mv", "Br", "An", "Ip", "Sb", "Wr", "Zh", "Kv", "Ka", "Em"]
        if character in allowed_lines:
            commands = []
            if character == "c":
                prefix = "<p>"
            else:
                prefix = "<d>"
            commands.append(prefix)
            if state['last_scene'] is not None and character != "c":
                commands.append("<scn>{}".format(state['last_scene']))
            commands.append('<msg>')                
            commands.append(character + " ")    
            commands.append("\"{}\"".format(clean_say_line(msg)))
            
            state['current_line'] += "".join(commands)       
            state['last_speaker'] = character

    def get_next_node(nodes, node_idx, cls, max_depth = 5):
        for i in range(node_idx, min(len(nodes), node_idx + max_depth)):
            if isinstance(nodes[i], cls):
                return nodes[i]
        return None

    def get_prev_node(nodes, node_idx, cls, max_depth = 5):
        for i in range(node_idx, max(0, node_idx - max_depth), -1):
            if isinstance(nodes[i], cls):
                return nodes[i]
        return None

    def copy_state_without_messages(state):
        obj = json.loads(json.dumps(state))
        obj['lines'] = []

    for node_idx, node in enumerate(nodes): 
        if isinstance(node, renpy.ast.Menu):
            forbidden_menu_items = [
                "Yes. I want to skip ahead.",
                "No. Don't skip ahead."
            ]
            next_say_node = get_next_node(nodes, node_idx, renpy.ast.Say)
            prev_say_node = get_prev_node(nodes, node_idx, renpy.ast.Say)
            for menu_item_idx, menu_item in enumerate(node.items):
                menu_str = menu_item[0].strip()
                if menu_str in forbidden_menu_items:
                    continue
                menu_str = re.sub(character_images_regex, r"\1", menu_str)
                menu_str = re.sub(regular_images_regex, r"", menu_str)
                if menu_item[2] is not None:
                    if menu_item_idx > 0 and prev_say_node is not None:
                        info = prev_say_node.diff_info()
                        process_say(info[1], info[2])
                        end_buffer(state)
                    process_say('c', menu_str)
                    end_buffer(state)
                    # Loop over menu item nodes
                    for node2 in menu_item[2]:
                        if isinstance(node2, renpy.ast.Say):
                            info = node2.diff_info()
                            process_say(info[1], info[2])
                            end_buffer(state)
                    if menu_item_idx < len(node.items) - 1 and next_say_node is not None:
                        info = next_say_node.diff_info()
                        process_say(info[1], info[2])
                        end_buffer(state)
        elif isinstance(node, renpy.ast.Label):
            extracted_label = extract_for_training(node.block, copy_state_without_messages(state))
        elif isinstance(node, renpy.ast.Say):
            should_end_buffer = False
            info = node.diff_info()
            process_say(info[1], info[2])
            end_buffer(state)
        elif isinstance(node, renpy.ast.If):
            filtered_nodes = []
            for entry in node.entries:
                for inner_node in entry[1]:
                    if isinstance(inner_node, renpy.ast.Say):
                        filtered_nodes.append(inner_node)
            if len(filtered_nodes) > 1:
                # More than 1 say node means we probably have a branch
                for entry in node.entries:
                    extracted_if_result = extract_for_training(entry[1], copy_state_without_messages(state))
                    state['lines'] += extracted_if_result
            elif len(filtered_nodes) > 0:
                info = filtered_nodes[0].diff_info()
                process_say(info[1], info[2])
        elif isinstance(node, renpy.ast.Scene):
            state['last_scene'] = node.diff_info()[1][0]
    return state['lines']

def parse():
    script_folder = os.path.dirname(os.path.realpath(__file__))
    awsw_path = os.path.join(script_folder, "..", "Angels with Scaly Wings", "game")
    # awsw_path = os.path.join(script_folder, "test_rpy")
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
                        result_data = "\n".join(result_data)
                        if result_data is not None:
                            training_data_fd.write(result_data)
                        result_sentiment = extract_sentiment(ast)
                        if result_sentiment is not None:
                            sentiment_data_fd.write(result_sentiment + "\n")