## install

```bash
pip install -r requirements.txt
```

## Generate table from audio

- put 151VNNVM.WAV in data/
- `python data_proc/split.py`
- `python parameters_extract/build_basic_table.py`

## Generate audio from notation and table

- put a notation image in data/notation/
- `python notation_to_wav/note_construct_basic.py`

