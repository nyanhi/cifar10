# coding=utf-8
import os
import time

root_logdir = os.path.join(os.curdir, "logs")  # moduleの名前か、modelの名前に合わせたdirectory名にしたい


def get_run_logdir(dir_name=""):
    run_time = time.strftime("_%Y_%m_%d-%H_%M_%S")
    run_id = dir_name + run_time
    return os.path.join(root_logdir, run_id)


