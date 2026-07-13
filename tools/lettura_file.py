import os
from color_logger import ColoreLog
import difflib

FILE_DIR = "/home/andrea/"

def get_all_files(subdir=None):
    """
    Recupera l'elenco di tutti i file presenti nella directory principale o in una sottodirectory specifica.

    Args:
        subdir (str, optional): Nome della sottodirectory da cui recuperare i file. Defaults to None.

    Returns:
        list: Elenco dei nomi dei file, ordinati alfabeticamente.
    """
    if not os.path.exists(FILE_DIR):
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Path inesistente.")
        return []

    if subdir is None:
        file_list = [
            f for f in os.listdir(FILE_DIR)
            if os.path.isfile(os.path.join(FILE_DIR, f))
        ]
        return sorted(file_list)
    else:
        file_subdir = os.path.join(FILE_DIR, subdir)
        if not os.path.exists(file_subdir):
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Path subdir inesistente.")
            return []
        else:
            file_list = [
                f for f in os.listdir(file_subdir)
                if os.path.isfile(os.path.join(file_subdir, f))
            ]
            return sorted(file_list)

def search_file(filename, subdir=None):
    """
    Cerca un file specifico nella directory principale o in una sottodirectory.

    Args:
        filename (str): Nome del file da cercare.
        subdir (str, optional): Nome della sottodirectory in cui cercare. Defaults to None.

    Returns:
        str: Percorso completo del file se trovato, altrimenti una lista di file simili.
    """
    file_list = get_all_files(subdir)

    if filename in file_list:
        if subdir is None:
            return os.path.join(FILE_DIR, filename)
        else:
            file_subdir = os.path.join(FILE_DIR, subdir)
            return os.path.join(file_subdir, filename)    
    else:
        return difflib.get_close_matches(filename, file_list, cutoff=0.6)

def read_file(file_path):
    """
    Legge il contenuto di un file specifico.

    Args:
        file_path (str): Percorso del file da leggere.

    Returns:
        str: Contenuto del file, oppure un messaggio di errore se il file non esiste.
    """
    if not os.path.exists(file_path):
        return "Path inesistente."
    else:    
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content

def mod_file(file_path, content):
    """
    Modifica o crea un file con il contenuto specificato.

    Args:
        file_path (str): Percorso del file da modificare o creare.
        content (str): Nuovo contenuto da inserire nel file.

    Returns:
        str: Messaggio di conferma che indica che il file è stato salvato correttamente.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File '{file_path}' salvato correttamente."

def print_file_list(file_list):
    """
    Stampa l'elenco dei file disponibili.

    Args:
        file_list (list): Elenco dei nomi dei file da stampare.
    """
    if file_list :
        print("\nFile presenti: ")
        for i, file in enumerate(file_list,1):
            print(f"{i} - {file}")
    else:
        print("\nNessun file presente")

def select_file(file_list):
    """
    Permette all'utente di selezionare un file dall'elenco fornito.

    Args:
        file_list (list): Elenco dei nomi dei file disponibili.

    Returns:
        str: Nome del file selezionato, o None se l'utente non ha selezionato alcun file.
    """
    print("\nVuoi selezionare un file specifico? S/N")
    user_input = input().strip()

    if user_input.lower() == "s":
        while True:
            print_file_list(file_list)
            print("\nQuale file vuoi selezionare?")
            user_input_int = input()

            if user_input_int.isdigit():
                i = int(user_input_int) - 1
                if 0 <= i < len(file_list):
                    return file_list[i]
                else:
                    print(f"{ColoreLog.WARNING}WARNING{ColoreLog.RESET}Numero inserito non valido, inserisci quello corretto.")
    else:
        print("Nessun file selezionato.")