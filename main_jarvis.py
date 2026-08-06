from model_select_jarvis import get_model_selection
from model_select_embedding import conferma_uso_emb, load_model_emb
from server_jarvis_IDE import JarvisServerIDE
from utilities.color_logger import ColoreLog
import sys
from tools.embedding import generate_embedding, select_file_for_emb, chunk_testo

def avvio_jarvis():

    funzioni = ["Chat", "Embedding",]

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

    print("\nBenvenuto nel programma di selezione delle IA del server Jarvis\n")

    while True:
        print("\nChe tipo di funzione vuoi che abbia Jarvis?\n")
        for i, f in enumerate(funzioni):
            print(f"{i+1} - {f}")
        try:
            user_input = input()

            if user_input == "1":
                model_name, model_path, model_type = get_model_selection()
                jarvis = JarvisServerIDE(model_name, model_path, model_type)
                jarvis.run_server_IDE()
            elif user_input == "2":
                model_name, model_path = conferma_uso_emb()
                emb_model, emb_tokenizer = load_model_emb(model_name, model_path)
                input_text = select_file_for_emb()
                chunks = chunk_testo(input_text, emb_tokenizer, 1000, 100)
                print(chunks)
                file_emb_test = generate_embedding(model_name, emb_model, emb_tokenizer, chunks)
                print(file_emb_test)
            elif user_input.lower() in ["exit", "esci"]:
                print(f"\n{ColoreLog.STOP}[STOP]{ColoreLog.RESET} Server Jarvis arrestato\n")
                sys.exit(0)
            else:
                print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Scelta non valida.")    
        except Exception as e:
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Inserire un numero valido.")
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore: {e}")
        

    
if __name__ == "__main__":

    avvio_jarvis()
