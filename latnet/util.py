import os
import random
import string

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def check_dir_exists(dir_name):
    """
    Checks if folder ``dir_name`` exists, and if it does not exist, it will be created.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    """generates a random sequence of character of length ``size``"""

    return ''.join(random.choice(chars) for _ in range(size))


def get_git():
    """
    If the current directory is a git repository, this function extracts the hash code, and current branch

    Returns
    -------
    hash : string
     hash code of current commit

    branch : string
     current branch
    """
    hash=None
    try:
        from subprocess import Popen, PIPE

        gitproc = Popen(['git', 'show-ref'], stdout = PIPE)
        (stdout, stderr) = gitproc.communicate()

        gitproc = Popen(['git', 'rev-parse',  '--abbrev-ref',  'HEAD'], stdout = PIPE)
        (branch, stderr) = gitproc.communicate()
        branch = branch.split('\n')[0]
        for row in stdout.split('\n'):
            if row.find(branch) != -1:
                hash = row.split()[0]
                break
    except:
        hash = None
        branch = None
    return hash, branch


def drange(start, stop, step):
    """
    Generates an array of floats starting from ``start`` ending with ``stop`` with step ``step``
    """
    r = start
    while r < stop:
        yield r
        r += step
