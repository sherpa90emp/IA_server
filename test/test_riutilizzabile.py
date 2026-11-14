import openvino_genai as ov_genai

# Interroga l'help nativo dei binding Python
#help(ov_genai.SchedulerConfig)

#help(ov_genai.GenerationConfig)

c = ov_genai.SchedulerConfig()
#print(c.to_string())

print(c.max_num_batched_tokens, c.max_num_seqs, c.block_size, c.cache_size, c.dynamic_split_fuse)