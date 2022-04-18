import os
import re
import glob

interactable_characters = {
    'Ry': 'Remy',
    'Lo': 'Lorem',
    'Ip': 'Ipsum',
    'Br': 'Bryce',
    'Em': 'Emera',
    'Ka': 'Katsuharu',
    'Rz': 'Reza',
    'Kv': 'Kevin',
    'Zh': 'Zhong',
    'Mv': 'Maverick',
    'An': 'Anna',
    'Ad': 'Adine',
    'Sb': 'Sebastian',
    'Nm': 'Naomi',
    'Iz': 'Izumi'
}
allowed_characters = list(interactable_characters.keys()) + ['c']

def parse():
    script_folder = os.path.dirname(os.path.realpath(__file__))
    awsw_path = os.path.join(script_folder, "..", "Angels with Scaly Wings", "game")
    # awsw_path = os.path.join(script_folder, "test_rpy")
    rpy_files = glob.glob(os.path.join(awsw_path, "*.rpy"))
    re_say_command = re.compile(r'([A-Za-z]{1,2})\s(.*?)"(.*?)"')
    re_scene_command = re.compile(r'scene\s([^ ]*)')
    re_menu_option = re.compile(r'"(.*?)":')
    with open("training_data.txt", 'w') as training_data_fd:
        with open("sentiment_training_data.txt", 'w') as sentiment_data_fd:
            blacklist = ["screens.rpy", "status.rpy", "help.rpy", "achievements.rpy", "gallery.rpy", "sec.rpy"]
            for rpy_file in rpy_files:
                file_name = os.path.basename(rpy_file)
                if not file_name in blacklist:
                    print(f"Parsing {rpy_file}")
                    last_scene = None
                    with open(rpy_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            scene_command_match = re_scene_command.match(line)
                            if scene_command_match is not None:
                                last_scene = scene_command_match.group(1)
                                continue
                            menu_option_match = re_menu_option.match(line)
                            if menu_option_match is not None:
                                forbidden_menu_items = [
                                    "Yes. I want to skip ahead.",
                                    "No. Don't skip ahead."
                                ]
                                menu_content = menu_option_match.group(1)
                                if not menu_content in forbidden_menu_items:
                                    msg_output = f'<p><msg>c "{menu_content}"'
                                    training_data_fd.write(msg_output + "\n")
                                continue
                            if last_scene is not None:
                                say_command_match = re_say_command.match(line)
                                if say_command_match is not None:
                                    msg_from = say_command_match.group(1)
                                    if msg_from in allowed_characters:
                                        msg_emote = say_command_match.group(2)
                                        msg_content = say_command_match.group(3)
                                        p_or_d = 'p' if msg_from == 'c' else 'd'
                                        scn_part = ''
                                        if msg_from != 'c':
                                            scn_part = f'<scn>{last_scene}'
                                        msg_output = f'<{p_or_d}>{scn_part}<msg>{msg_from} "{msg_content}"'
                                        training_data_fd.write(msg_output + "\n")
if __name__ == "__main__":
    parse()