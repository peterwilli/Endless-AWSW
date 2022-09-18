import sys
from regexes import *
from collections import Counter
from typing import List
import math

def count_characters(lines) -> Counter:
    character_batch_counter = Counter()
    for line in lines:
        msg_match = re_msg.search(line)
        if msg_match is None:
            raise Exception(f"msg_match None! Line: '{line}'")
        msg_from = msg_match.group(1)
        character_batch_counter[msg_from] += 1
    return character_batch_counter

def spread(lines, character_counts) -> List[str]:
    character_with_most_screen_time = max(character_counts, key=character_counts.get)
    result = []
    last_c = []
    for line in lines:
        msg_match = re_msg.search(line)
        if msg_match is None:
            raise Exception(f"msg_match None! Line: '{line}'")
        msg_from = msg_match.group(1)
        if msg_from == "c":
            last_c.append(line)
        msg_from_screentime = character_counts[msg_from]
        screen_time_multiplier = math.floor(character_counts[character_with_most_screen_time] / msg_from_screentime)
        for i in range(screen_time_multiplier):
            result.append(line)
    return result

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, "r") as f:
        lines = list(filter(lambda line: len(line.strip()) > 0, f.readlines()))
        character_counts = count_characters(lines)
        result = spread(lines, character_counts)
        with open(output_file, "w") as fw:
            for line in result:
                fw.write(line)