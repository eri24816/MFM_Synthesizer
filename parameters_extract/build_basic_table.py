from src.tableGeneration import tableGeneration
from pathlib import Path
instr = 'cello'
genre = 'sustain'
pitch_start = 36
pitch_end = 81

tg = tableGeneration()
# if instr == 'cello':
#     tg.max_freq = 2000

for pitch in range(pitch_start,pitch_end+1):
    audio_path = Path(f'data/{instr}/{genre}/audio/{pitch}.wav')
    table_path = Path(f'data/{instr}/{genre}/table/{pitch}.npz')
    audio_path.parent.mkdir(parents=True,exist_ok=True)
    table_path.parent.mkdir(parents=True,exist_ok=True)
    tg.implement(audio_path, table_path, pitch, max_n_partial=30)
        
