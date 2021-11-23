import sys
import pyximport; pyximport.install()
sys.path.append('/usr/share/games/renpy')
import renpy
import renpy.display
import renpy.ast as ast
from renpy import parser
import traceback

# Shitty hacks to make renpy global var still appear and working.
class Config:
    basedir = "/src/extractor"
    renpy_base = "/usr/share/games/renpy"
    log_enable = True

class Files:
    def add(self, fname):
        pass

class ScriptEdit:
    files = Files()
    def Line(self, fname, num1, num2):
        pass

class Game: 
    exception_info = ""

    def context(self):
        class Context:
            init_phase = False
        return Context

renpy.config = Config
renpy.game = Game()
renpy.scriptedit = ScriptEdit()

try:
    print(parser.parse("/src/awsw/game/adine3.rpy"))
except:
    print(traceback.format_exc())