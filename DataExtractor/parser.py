from renpy import parser
import os

def parse():
    script_folder = os.path.dirname(os.path.realpath(__file__))
    ast = parser.parse(os.path.join(script_folder, "../Angels with Scaly Wings/game/adine3.rpy"))
    print(ast)