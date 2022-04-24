import os

class Config:
    base_model_name = "distilgpt2"
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