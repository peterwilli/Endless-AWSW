import re
import logging
import lmql
import nest_asyncio
nest_asyncio.apply()
import sys
from multiprocessing import Process

class ModelManager:
    def __init__(self, model_id = None):
        self.max_length = 128
        self.reply_prefix = "<d><scn>"
        self.model_id = model_id
        self.load_model()

    @lmql.query
    async def awsw_reply(self, past, prompt):
        '''lmql
        sample(25, 0.8)
        "{past}<p><msg>c \\\"{prompt}\\\"<d><scn>[SCENE]<msg>[DRAGON] [EMOTION] \\\"[REPLY]\\\"" \
        where SCENE in ['ecknaomiapt01', 'eckoldbiolab', 'beach', 'eckwatersky01', 'eckunderwatertransition01', 'eckunderwater02', 'eckunderwatertunnel', 'black', 'eckbeachb', 'park2', 'loremapt', 'office', 'bare', 'pad', 'bareblur', 'bareblur2', 'facin2', 'facinx', 'facin3', 'alley', 'fb', 'farm', 'town4', 'adineapt', 'emeraroom', 'o4', 'o2', 'park3', 'np3x', 'np2x', 'np1x', 'buildingoutside', 'np3', 'np2', 'store2', 'town1x', 'forestx', 'cave', 'o', 'remyapt', 'cafe', 'viewingspot', 'np1r', 'hallway', 'np2y', 'np1n', 'town2', 'darker', 'store', 'library', 'forest2', 'school', 'forest1', 'storex', 'np5e', 'beachx', 'padx', 'np4', 'np5', 'fac1', 'facin', 'town3', 'kitchen', 'np1', 'stars', 'o3', 'town7', 'town6', 'deadbody', 'whiteroom', 'cave2', 'table', 'starsrx', 'farm2', 'office2', 'hatchery', 'testingroom', 'gate', 'fac12', 'adineapt2', 'eckkitchenx', 'eckoutsideuniv2', 'eckswimmingpool', 'ecknewtownout2', 'eckswimmingpool2', 'eckpolicedeptstairs1', 'eckplayeraptextra1', 'town1', 'ecknaomiapt03', 'ecknaomiaptbalcony', 'ecknaomiapt02'] and \
            DRAGON in ['Nm', 'Ry', 'Lo', 'Ip', 'Br', 'An', 'Mv', 'Ad', 'Em', 'Sb', 'Dm', 'Ka', 'Rz', 'Iz', 'Zh', 'Kv'] and \
            EMOTION in ['smile', 'normal', 'shy', 'sad', 'blank', 'stern', 'concern', 'slsmile', 'confused', 'scared', 'annoyed', 'hurt', 'cry', 'look', 'happy', 'relieved', 'think', 'smirk', 'brow', 'laugh', 'flirty', 'face', 'disgust', 'rage', 'giggle', 'disappoint', 'mean', 'frown', 'ques', 'drop', 'disapproval', 'arrogant', 'exhausted', 'angry', 'gunpoint', 'gunself', 'amused', 'nice', 'serv', 'despair', 'defeat', 'ramble', 'hand', 'frustrated', 'excited', 'bacon', 'crysmile', 'sleep', 'sideeye'] and \
            STOPS_BEFORE(REPLY, '"')
        from
            lmql.model("/Projects/Personal/Endless-AWSW/Research/merged-eawsw-16k", endpoint="localhost:9999", tokenizer="lmsys/vicuna-7b-v1.5-16k")
        '''
        
    def load_model(self):
        thread = Process(target=lambda: lmql.serve(self.model_id, static=True, cuda=True, port=9999, trust_remote_code=True, load_in_8bit=True))
        thread.start()
        
    async def say(self, past, prompt) -> str:
        reply = await self.awsw_reply(past, prompt)
        return reply.prompt.rsplit('<d>', 1)[1].strip()