import os

class Config:
    base_model_name = "EleutherAI/gpt-neo-125M"
    base_model_basename = base_model_name.split("/")[1]
    work_dir = os.path.join("/opt", "awsw")
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
        'Nm': 'Naomi'
    }
    