from sys import platform
from os import system

def clear():
    """ Clears console """

    if platform == 'win32':
        system('cls')
    else:
        system('clear')


def wait_for_enter():
    input('Press Enter to continue...')
    clear()
