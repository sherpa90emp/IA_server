from colorama import init, Fore

init(autoreset=True)

class ColoreLog:
    RESET = Fore.RESET
    ERRORE = Fore.RED
    INFO = Fore.BLUE
    WARNING = Fore.YELLOW
    SUCCESS = Fore.GREEN
    DEBUG = Fore.CYAN
    STOP = Fore.MAGENTA
    TOOL = Fore.LIGHTGREEN_EX
    INPUT = Fore.LIGHTBLUE_EX