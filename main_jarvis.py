from model_select_jarvis import get_model_selection, load_draft_model
from model_select_embedding import conferma_uso_emb, load_model_emb
from server_jarvis_IDE import JarvisServerIDE
from utilities.color_logger import ColoreLog
import sys
from tools.embedding import generate_embedding, select_file_for_emb, chunk_testo

def avvio_jarvis():
    """
    Funzione principale per l'avvio del server Jarvis. Gestisce la selezione del tipo di funzione
    (Chat o Embedding) e avvia il rispettivo processo.
    
    Funzionalità:
    - Visualizza il logo e il messaggio iniziale
    - Gestisce la scelta dell'utente tra Chat e Embedding
    - Avvia il server Chat (JarvisServerIDE) o esegue il processo di embedding
    - Gestisce l'uscita dal programma
    
    La funzione utilizza un ciclo infinito per gestire l'input dell'utente e gestisce eccezioni
    per input non validi.
    """

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
        print(f"\n{ColoreLog.INPUT}[INPUT]{ColoreLog.RESET}Che tipo di funzione vuoi che abbia Jarvis?\n")
        for i, f in enumerate(funzioni):
            print(f"{i+1} - {f}")
        try:
            user_input = input()

            if user_input == "1":
                """
                Avvia il server Chat:
                1. Seleziona il modello con get_model_selection()
                2. Crea un'istanza di JarvisServerIDE
                3. Avvia il server con run_server_IDE()
                """

                model_name, model_path, model_type = get_model_selection()
                model_draft, model_draft_path = load_draft_model()
                jarvis = JarvisServerIDE(model_name, model_path, model_type, model_draft, model_draft_path)
                jarvis.run_server_IDE()

            elif user_input == "2":
                """
                Processo di Embedding:
                1. Conferma l'uso del modello di embedding con conferma_uso_emb()
                2. Carica il modello e il tokenizer con load_model_emb()
                3. Seleziona il file da processare con select_file_for_emb()
                4. Suddivide il testo in chunk con chunk_testo()
                5. Genera gli embedding con generate_embedding()
                """

                model_name, model_path = conferma_uso_emb()
                emb_model, emb_tokenizer = load_model_emb(model_name, model_path)
                input_text = select_file_for_emb()
                chunks = chunk_testo(input_text, emb_tokenizer, 1000, 100)
                print(chunks)
                file_emb_test = generate_embedding(model_name, emb_model, emb_tokenizer, chunks)
                print(file_emb_test)

            elif user_input.lower() in ["exit", "esci"]:
                """
                Uscita dal programma:
                - Stampa un messaggio di arresto
                - Chiude il server
                - Esce dal programma
                """

                print(f"\n{ColoreLog.STOP}[STOP]{ColoreLog.RESET} Server Jarvis arrestato\n")
                sys.exit(0)

            else:
                print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Scelta non valida.")
                    
        except Exception as e:
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Inserire un numero valido.")
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore: {e}")
        

    
if __name__ == "__main__":
    """
    Punto di ingresso principale del programma.
    Chiama la funzione avvio_jarvis() per iniziare l'esecuzione del server Jarvis.
    """

    avvio_jarvis()
