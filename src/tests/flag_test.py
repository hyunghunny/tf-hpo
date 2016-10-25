#!/usr/bin/python

"""tensorflow flags management test.

Set the following flags to set the parameters in use:
--set_bool={True or False}
--set_string={a string value}
--set_int={an integer value}
--set_float={a float point value}
"""
from __future__ import print_function
import sys
import traceback
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def define_flags():
    try:
        tf.app.flags.DEFINE_boolean("set_bool", False, "boolean flag test")
        tf.app.flags.DEFINE_string("set_string", "None", "string flag test")
        tf.app.flags.DEFINE_integer("set_int", 0, "integer flag test")
        tf.app.flags.DEFINE_float("set_float", .0, "float flag test")
    except:
        print("invalid argument setting! See below:")
        print (__doc__)
        e = sys.exc_info()[0]
        traceback.print_exc()
        print (e)
        sys.exit(0)

    tf.app.flags.DEFINE_boolean("done", True, "check the flag configuration")

# check the flags have been configured    
try:    
    if FLAGS.done is False:        
        define_flags()
except:
    define_flags()
    
    
def main(argv=None):
        

    print("boolean value: " + str(FLAGS.set_bool))
    print("string value: " + str(FLAGS.set_string))
    print("integer value: " + str(FLAGS.set_int))
    print("float value: " + str(FLAGS.set_float))

    sys.exit(0)

if __name__ == "__main__":
    tf.app.run()
