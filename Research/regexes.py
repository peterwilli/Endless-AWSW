import re

re_token = re.compile(r'(<.*?>|[^<]*)')
re_command = re.compile(r'^<(.*?)>$')
re_msg = re.compile(r'([A-Za-z]{1,2})\s(.*?)\s{0,1}"(.*)"')