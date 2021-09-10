import glob
import sys
import os

awsw_path = sys.argv[1]
rpy_files = glob.glob(os.path.join(awsw_path, "*.rpy"))

with open("training_data.txt", 'w') as training_data_fd:
    for file in rpy_files:
        with open(file, 'r') as fd:
            content = fd.read()
            lines = content.split("\n")
            for line in lines:
                allowed_lines = ["m", "Rz", "Lo", "Ad", "c", "Ry", "Br", "An", "Sb", "Wr", "Zh", "Kv", "Ka"]
                line_stripped = line.strip()
                split_lines = line_stripped.split(" ")
                if split_lines[0] in allowed_lines:
                    training_data_fd.write(line_stripped + "\n")