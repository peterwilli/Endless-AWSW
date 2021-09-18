import glob
import sys
import os
import re

awsw_path = sys.argv[1]
rpy_files = glob.glob(os.path.join(awsw_path, "*.rpy"))

with open("training_data.txt", 'w') as training_data_fd:
    for file in rpy_files:
        with open(file, 'r') as fd:
            content = fd.read()
            lines = content.split("\n")
            in_menu = False
            menu_item_regex = re.compile(r'\"(.*?)\":')
            out_of_block_regex = re.compile(r'[^\s\t]')
            action_menu_regex = re.compile(r'\[\[.*?\]')
            character_images_regex = re.compile(r'{#.*?}(.*)')
            regular_images_regex = re.compile(r'{image=.*?}')
            menu_replies = []
            for line in lines:
                allowed_lines = ["n", "m", "Rz", "Lo", "Ad", "c", "Ry", "Mv", "Br", "An", "Sb", "Wr", "Zh", "Kv", "Ka"]
                line_stripped = line.strip()

                # Parse edge case with menu (convert it to "C")
                if line_stripped == "menu:":
                    in_menu = True
                elif in_menu:
                    if menu_item_regex.match(line_stripped):
                        menu_str = re.findall(menu_item_regex, line_stripped)[0]
                        menu_str = re.sub(character_images_regex, r"\1", menu_str)
                        menu_str = re.sub(regular_images_regex, r"", menu_str)
                        menu_str = menu_str.strip()
                        forbidden_menu_items = [
                            "Yes. I want to skip ahead.",
                            "No. Don't skip ahead."
                        ]
                        if not action_menu_regex.match(menu_str):
                            if not menu_str in forbidden_menu_items:
                                training_data_fd.write(f'c "{menu_str}"\n')
                    elif len(line) > 0 and out_of_block_regex.match(line[0]):
                        in_menu = False
                        
                split_lines = line_stripped.split(" ")
                if split_lines[0] in allowed_lines:
                    with_index = line_stripped.find('" with')
                    if with_index > -1:
                        training_data_fd.write(line_stripped[:with_index + 1] + "\n")
                    else:
                        training_data_fd.write(line_stripped + "\n")