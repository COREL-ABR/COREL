import subprocess
from subprocess import CalledProcessError


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def run_shell_cmd(cmd):
    print(f"{bcolors.HEADER}RUNNING COMMAND: {cmd}{bcolors.ENDC}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    for line in iter(process.stdout.readline, b''):
        print('| ' + line.decode('utf-8').rstrip())

    output = process.communicate()[0]

    if process.returncode != 0:
        raise CalledProcessError(process.returncode, cmd)

    return output
