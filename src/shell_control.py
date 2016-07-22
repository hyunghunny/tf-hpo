#!/usr/bin/python

import spur

def connect_shell() :

    try:
        shell = spur.SshShell(hostname="localhost", \
                              username="webofthink", password="joyan1029", \
                             missing_host_key=spur.ssh.MissingHostKey.accept)

        with shell:
            #result = shell.run("sh", "-c", "python ./tf-hpo/src/cnn_layer2_multi.py -h")
            process = shell.spawn("sh", "-c", "echo hello")
            result = process.wait_for_result()
            print result.output
    except spur.ssh.ConnectionError as error:
        print error.original_traceback
        raise

if __name__ == "__main__":
    connect_shell()
