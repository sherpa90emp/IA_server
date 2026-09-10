import openvino_genai as ov_genai

# Interroga l'help nativo dei binding Python
help(ov_genai.SchedulerConfig)

# Oppure crea un'istanza e stampa la sua rappresentazione stringa
config = ov_genai.SchedulerConfig()
print(config.to_string())