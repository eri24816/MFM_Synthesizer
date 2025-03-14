import librosa
import numpy as np
def detect_onsets(audio,n_fft=64,visualize=False):
    '''
    Detects onsets in audio using a simple thresholding method.
    Returns sample indices of onsets.
    '''
    hop = n_fft//4
    mag = np.abs(librosa.stft(audio,n_fft=n_fft,hop_length=hop)).mean(axis=0)

    def shift(x,n):
        if n > 0:
            return np.concatenate([[False]*n,x[:-n]])
        else:
            return np.concatenate([x[-n:], [False]*(-n)])
    on_thres = 0.001
    on_local = mag > on_thres
    on = on_local.copy()
    for s in [1,2,3,10,100,200,300]:
        on &= shift(on_local,-s)

    off_thres = 0.0005
    off_local = mag < off_thres
    off = off_local.copy()
    for s in [1,2,3,10,50,100]:
        off &= shift(off_local,-s)

    on_positive_edge = np.diff(on) > 0
    off_positive_edge = np.diff(off) > 0


    onsets = []
    on_positive_edge_pos = np.where(on_positive_edge)[0]
    off_positive_edge_pos = np.where(off_positive_edge)[0]

    on_cursor = 0
    off_cursor = 0
    current_off = -1
    while True:
        # find the onset after the current off
        while on_cursor < len(on_positive_edge_pos) and on_positive_edge_pos[on_cursor] <= current_off:
            on_cursor += 1
        if on_cursor >= len(on_positive_edge_pos):
            break
        current_on = on_positive_edge_pos[on_cursor]
        onsets.append(current_on)
        
        # find the off after the on
        while off_cursor < len(off_positive_edge_pos) and off_positive_edge_pos[off_cursor] <= current_on:
            off_cursor += 1
        if off_cursor >= len(off_positive_edge_pos):
            break
        current_off = off_positive_edge_pos[off_cursor]

    if visualize:
        from matplotlib import pyplot as plt
        start = 5040000
        stop = start+80000
        n = len(np.arange(0,stop-start-hop,hop))
        plt.plot(audio[start:stop]*2)
        plt.plot(np.arange(0,stop-start-hop,hop),mag[start//hop:start//hop+n])
        plt.plot(np.arange(0,stop-start-hop,hop),0.02*on_positive_edge[start//hop:start//hop+n])
        plt.plot(np.arange(0,stop-start-hop,hop),-0.02*off_positive_edge[start//hop:start//hop+n])
        plt.vlines(np.array(onsets) * hop-start,0,0.08,color='black')
        plt.xlim(0,stop-start)
        plt.show(block=True)

    return np.array(onsets) * hop


input_path = r'.\data\151VNNVM.WAV'
output_path = r'.\data\violin\sustain\audio'

if __name__ == '__main__':
    from pathlib import Path 
    import soundfile as sf


    audio, sr = librosa.load(input_path)
    onsets = detect_onsets(audio,n_fft=64,visualize=True)
    
    segments = np.split(audio,onsets)[1:]
    
    # trim end of segments
    segments = [s[:-4000] for s in segments]
    
    # save segments
    Path(output_path).mkdir(parents=True,exist_ok=True)

    # violin
    for i,s in enumerate(segments[0:13] + segments[19:26] + segments[32:39] + segments[45:]):
        sf.write(f'{output_path}/{55+i}.wav',s,sr)

    # cello
    # for i,s in enumerate(segments[0:7] + segments[13:20] + segments[26:33] + segments[39:]):
    #     sf.write(f'{output_path}/{36+i}.wav',s,sr)