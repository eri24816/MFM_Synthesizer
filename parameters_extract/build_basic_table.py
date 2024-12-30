from src.tableGeneration import *
import os 

instr = 'cello'
genre = 'staccato_m'
# genre = 'long'

TG = tableGeneration()

for note in range(49, 58):
# for note in range(67,80):
    seq = 1
    while os.path.exists(f'../original_file/{instr}/{genre}/note{note}-{seq}.wav'):
        TG.implement(instr, genre, note, seq)
        seq += 1
        
