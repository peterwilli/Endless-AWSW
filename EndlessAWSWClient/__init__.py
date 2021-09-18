from modloader.modclass import Mod, loadable_mod
from modloader import modast, modinfo

@loadable_mod
class AWSWMod(Mod):
    def mod_info(self):
        return ("EndlessAWSW", "1.0", "EmeraldOdin")

    def mod_load(self):
        ml = modinfo.get_mods()["MagmaLink"].import_ml()
        ml.find_label("seccont") \
            .search_say("He seemed a lot smaller than Remy, and when he somewhat") \
            .hook_to("pick_your_poison") \
            .search_say("Just follow me.") \
            .link_from("pick_your_poison_end")


    def mod_complete(self):
        pass
