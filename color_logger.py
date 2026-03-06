from colorama import init, Fore

init(autoreset=True)

class ColoreLog:
    ERRORE = Fore.RED
    RESET = Fore.RESET
    INFO = Fore.BLUE
    WARNING = Fore.YELLOW
    SUCCESS = Fore.GREEN