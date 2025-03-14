from src.tableGeneration import tableGeneration
from pathlib import Path
instr = 'violin'
genre = 'sustain'
pitch_start = 55
pitch_end = 100

tg = tableGeneration()

for pitch in range(pitch_start,pitch_end+1):
    audio_path = Path(f'data/{instr}/{genre}/audio/{pitch}.wav')
    table_path = Path(f'data/{instr}/{genre}/table/{pitch}.npz')
    audio_path.parent.mkdir(parents=True,exist_ok=True)
    table_path.parent.mkdir(parents=True,exist_ok=True)
    tg.implement(audio_path, table_path, pitch, max_n_partial=30)
        
