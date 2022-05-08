import os

class Config:
    base_model_name = "EleutherAI/gpt-neo-125M"
    base_model_basename = base_model_name.split("/")[1]
    work_dir = os.path.join("/opt", "awsw")
    interactable_characters = {
        "Ad": "Adine",
        "An": "Anna",
        "Br": "Bryce",
        "Dm": "Damion",
        "Em": "Emera",
        "Ip": "Ipsum",
        "Iz": "Izumi",
        "Ka": "Katsuharu",
        "Kv": "Kevin",
        "Lo": "Lorem",
        "Mv": "Maverick",
        "Nm": "Naomi",
        "Ry": "Remy",
        "Rz": "Reza",
        "Sb": "Sebastian",
        "Zh": "Zhong",
    }
