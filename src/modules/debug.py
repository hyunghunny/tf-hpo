from __future__ import print_function

# LOG LEVEL
SHOW_DEBUG = False
SHOW_ERR = True

def debug(*args):
    if SHOW_DEBUG:
        print(args)
        
def error(*args):
    if SHOW_ERR:
        print(args)
