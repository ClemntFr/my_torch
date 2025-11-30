import json

with open("chess_db2.json", "r") as f:
    fens = json.load(f)

print(fens["check white"][:5])