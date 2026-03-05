from model_select_jarvis import get_model_selection
from server_jarvis_IDE import JarvisServerIDE

def avvio_jarvis():
    print("---------------------------------------------------------")
    print("      _    ______    ______   __        __   _    _____  ")
    print("     | |  |  __  |  |  __  |  \ \      / /  | |  |  ___| ")
    print("     | |  | |  | |  | |__| |   \ \    / /   | |  | |___  ")
    print("     | |  | |__| |  |    __|    \ \  / /    | |  |___  | ")
    print("  _  | |  |  __  |  |  _ \       \ \/ /     | |      | | ")
    print(" | |_| |  | |  | |  | | \ \       \  /      | |   ___| | ")
    print(" |_____|  |_|  |_|  |_|  \_\       \/       |_|  |_____| ")
    print("                                                         ")
    print("---------------------------------------------------------")

    print("\nBenvenuto nel programma di selezione delle IA del server Jarvis")

    
if __name__ == "__main__":

    avvio_jarvis()
    
    name, path = get_model_selection()

    jarvis = JarvisServerIDE(name, path)

    jarvis.run_server_IDE()