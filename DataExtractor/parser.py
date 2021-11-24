import renpy
import os
import json

def extract_for_training(nodes, clamp_branch = False):
    result = []
    buffer = []
    last_speaker = None
    for node in nodes:
        if isinstance(node, renpy.ast.Menu):
            for menu_item in node.items:
                buffer.append("c \"{}\"".format(menu_item[0]))
                buffer.append(extract_for_training(menu_item[2], True))
        if isinstance(node, renpy.ast.Say):
            info = node.diff_info()
            if info[1] == "c" and last_speaker != "c":
                result.append(" ".join(buffer))
                buffer = [
                    "{} \"{}\"".format(info[1], info[2])
                ]
            else:
                buffer.append("{} \"{}\"".format(info[1], info[2]))
            last_speaker = info[1]
    # add any excess replies to the last line...
    result.append(" ".join(buffer))
    sep = "\n"
    if clamp_branch:
        sep = " "
    return sep.join(result)

def parse():
    script_folder = os.path.dirname(os.path.realpath(__file__))
    ast = renpy.parser.parse(os.path.join(script_folder, "../Angels with Scaly Wings/game/adine3.rpy"))
    print(extract_for_training(ast))