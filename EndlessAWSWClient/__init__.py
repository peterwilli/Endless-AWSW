from modloader.modclass import Mod, loadable_mod
from modloader import modast, modinfo
import jz_magmalink as ml

@loadable_mod
class AWSWMod(Mod):
    name = "EndlessAWSW"
    version = "0.1"
    author = "EmeraldOdin"
    dependencies = ["MagmaLink"]

    def mod_load(self):
        modast.set_renpy_global("eawsw_naomi_installed", modinfo.has_mod("A Solitary Mind"))
        ml.find_label("seccont") \
            .search_say("He seemed a lot smaller than Remy, and when he somewhat") \
            .hook_to("eawsw_intro") \
            .search_say("Just follow me.") \
            .link_from("pick_your_poison_end")

    def mod_complete(self):
        pass