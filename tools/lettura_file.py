import os

def serch_file():
    file_dir = "/home/andrea/file_IA"
    if not os.path.exists(file_dir):
        return []

    file_list = [
        f for f in os.listdir(file_dir)
        if os.path.isdir(os.path.join(file_dir, f))
    ]
    
    return sorted(file_list)
        

def select_file(file_list):
    if file_list:
        print("File prsenti")