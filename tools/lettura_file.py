import os
from color_logger import ColoreLog
import difflib

FILE_DIR = "/home/andrea/file_IA"

def get_all_files(subdir=None):
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
    if not os.path.exists(file_path):
        return "Path inesistente."
    else:    
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return content

def print_file_list(file_list):
    if file_list :
        print("\nFile presenti: ")
        for i, file in enumerate(file_list,1):
            print(f"{i} - {file}")
    else:
        print("\nNessun file presente")

def select_file(file_list):
    print("\nVuoi selezionare un file specifico? S\N")
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
    