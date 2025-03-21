import cv2
from notation_to_parameters import StrokeInfo, notation_to_parameters
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import iirfilter, filtfilt
plot = False

full_fig_len = 2.4 # sec
quick_atk = True
# atk_peak = 1.4
atk_peak = 1.3
release_len = 4000
intensity_factor = 0.3
envelope_factor = 0.02


thres = 0.5

timbre_gain = 1.4
shelving_filter_gain = 1.4



def timesfilt(b, a, signal, times):
    output = signal
    for _ in range(round(times / 2)):
        output = filtfilt(b, a, output)
    return output
    
def poly_fit(y, d=20):
    t = np.arange(y.shape[0]) / y.shape[0]
    z = np.polyfit(t, y, deg=d)
    p = np.poly1d(z)
    res = y - p(t)
    plt.plot(y)
    plt.plot(p(t))
    if plot:
        plt.show()
        plt.plot(res)
        plt.show()
    return p(t), res
    
def linearResample(y, target_sr):
    l = y.shape[0]
    tmp = librosa.resample(y, orig_sr=l, target_sr=(l * 100), res_type='linear')
    out = librosa.resample(tmp, orig_sr=tmp.shape[0], target_sr=target_sr, res_type='linear')
    if out.shape[0] != target_sr:
        out = out[:target_sr]
    return out

def concat(atk, sus, ovlp, mod='hann'):
    split = [0] * 4
    split[1] = atk.shape[0]
    split[0] = split[1] - ovlp
    
    length = atk.shape[0] + sus.shape[0] - ovlp
    out = np.zeros(shape=(length,))
    lin_env = np.arange(ovlp) / ovlp
    out[:split[0]] = atk[:split[0]]
    if mod == 'linear':
        out[split[0]:split[1]] = atk[split[0]:] * (1 - lin_env) + sus[:ovlp] * lin_env
    elif mod == 'hann':
        end_env = 0.5 * (1 + np.cos(np.pi*lin_env))
        start_env = 0.5 * (1 + np.cos(np.pi*(1 - lin_env)))
        out[split[0]:split[1]] = atk[split[0]:] * end_env + sus[:ovlp] * start_env
    out[split[1]:] = sus[ovlp:]
    return out


def release_decay(freq):
    # simple mode
    const_exp = 0.998
    var_exp = 1 - const_exp
    return const_exp + var_exp * np.power(1 - freq / 22050, 3)

'''
old stroke info:
[left_to_right_intensity, left_to_right_note, left_to_right_density, 
            left_to_right_hue, left_to_right_saturation, left_to_right_value, effective_x]
            0: intensity
            1: pitch
            2: density
            3: hue
            4: saturation
            5: value
            6: effective_x
'''

def note_construct(stroke_info: StrokeInfo, w, h, note, des):
    note_trend = (np.array(stroke_info.parameters['pos_y']) / h) - 0.5
    r = (np.max(note_trend) - np.min(note_trend))
    if  r > 1:
        note_trend = (note_trend - round(np.average(note_trend))) / r
    note_trend += note
    pitch = 440 * np.power(2, (note_trend-69)/12)
    
    table_path = f'data/{instr}/sustain/table/{note}.npz'

    pars = np.load(table_path)
    fs = int(pars['sampleRate'])
    fs_data = int(pars['par_sr'])

    note_len = pars['ori_sec']
    note_len = full_fig_len
    length = round(stroke_info.end_x/w * note_len * fs) - round(stroke_info.start_x/w * note_len * fs)
    pars_len = round(pars['sampleRate'] * pars['ori_sec'])
    start = round(stroke_info.start_x/w * note_len * fs) + pars['attackLen'] - pars['overlapLen']
    end = round(stroke_info.end_x/w * note_len * fs)
    length_sus = end - start
    
    t = np.arange(length) / fs
    t_sus = t[(pars['attackLen']-pars['overlapLen']):]
    
    # pitch and vibration curve
    pitch = linearResample(pitch, length)
    b, a = iirfilter(4, Wn=20, fs=fs, ftype="butter", btype="lowpass")
    pitch = timesfilt(b, a, pitch, 8)
    pitch, vib = poly_fit(pitch, d=5)
    vib = vib[(pars['attackLen']-pars['overlapLen']):] * 2
    
    intensity = linearResample(np.array(stroke_info.parameters['intensity']), length)
    intensity_sus = intensity[(pars['attackLen']-pars['overlapLen']):]
    density = linearResample(np.array(stroke_info.parameters['density']), length)
    density_sus = density[(pars['attackLen']-pars['overlapLen']):]
    hue = linearResample(np.array(stroke_info.parameters['hue']), length)
    hue_sus = hue[(pars['attackLen']-pars['overlapLen']):]
    saturation = linearResample(np.array(stroke_info.parameters['saturation']), length)
    saturation_sus = saturation[(pars['attackLen']-pars['overlapLen']):]
    value = linearResample(np.array(stroke_info.parameters['value']), length)
    value_sus = value[(pars['attackLen']-pars['overlapLen']):]
    
    # hue modify bow position
    hue_sus = np.clip(hue_sus, 0, 135)
    bow_pos = 1 / (hue_sus / 135 * 5 + 2)
    # bow_pos = 1 / (hue_sus / 135 * 10 + 2)
    
    base_freq = 440 * np.power(2, (note-69)/12)
    length_sus = length - pars['attackLen'] + pars['overlapLen']
    
    if (length_sus <= 0):
        return
    
    mixtone = np.zeros(shape=(length,))
    
    noise1, _ = librosa.load('parameters_extract/colored_noise.wav', sr=fs * (2000 / pars['coloredCutoff1']))
    noise2, _ = librosa.load('parameters_extract/colored_noise.wav', sr=fs * (2000 / pars['coloredCutoff2']))
    
    for partial in range(1, pars['partialAmount']+1):
        over_freq = base_freq * partial
        
        noise_fac = 0.15 * (1-density_sus)
        alpha_fac = 1.0
        mag_fac = 1.0
        
        
        alpha_attack = np.array(pars['alphaAttack'][partial-1])
        mag_attack = np.array(pars['magAttack'][partial-1])
            
        
        ag = np.array(pars['alphaGlobal'][partial-1])
        alpha_global = linearResample(ag, pars_len)
        alpha_global = alpha_global[start:end]
        pars['alphaLocal.env'][partial-1][0][0] = (pars['alphaLocal.env'][partial-1][0][1] + pars['alphaLocal.env'][partial-1][0][2]) / 2
        pars['alphaLocal.env'][partial-1][1][0] = (pars['alphaLocal.env'][partial-1][1][1] + pars['alphaLocal.env'][partial-1][1][2]) / 2
        env1 = linearResample(np.array(pars['alphaLocal.env'][partial-1][0]), pars_len)
        env2 = linearResample(np.array(pars['alphaLocal.env'][partial-1][1]), pars_len)
        env1 = env1[start:end]
        env2 = env2[start:end]
        c1 = pars['alphaLocal.spreadingCenter'][partial-1][0]
        c2 = pars['alphaLocal.spreadingCenter'][partial-1][1]
        fac1 = pars['alphaLocal.spreadingFactor'][partial-1][0]
        fac2 = pars['alphaLocal.spreadingFactor'][partial-1][1]
        ng1 = pars['alphaLocal.noiseGain'][partial-1][0]
        ng2 = pars['alphaLocal.noiseGain'][partial-1][1]
        noise_start = round(np.random.rand() * 1e+5)
        phase1 = noise1[noise_start:(noise_start+length_sus)]
        noise_start = round(np.random.rand() * 1e+5)
        phase2 = noise2[noise_start:(noise_start+length_sus)]
        alpha_local = np.sin(2*np.pi*c1*t_sus + fac1 * phase1) * env1 * ng1 \
            + np.sin(2*np.pi*c2*t_sus + fac2 * phase2) * env2 * ng2
        alpha_sus = alpha_global + pars['alphaLocal.gain'][partial-1] * alpha_local * noise_fac * alpha_fac
        alpha_attack -= (np.mean(alpha_attack[-pars['overlapLen']:]) - np.mean(alpha_sus[:pars['overlapLen']]))
        
        envelope = linearResample(np.array(pars['totalEnv']), pars_len)
        envelope = envelope[start:end]
        envelope = intensity_sus * intensity_factor + envelope * envelope_factor
        mag_ratio = linearResample(np.array(pars['magRatio'][partial-1]), pars_len)
        mag_ratio = mag_ratio[start:end]
        mag_global = envelope * mag_ratio
        pars['magLocal.env'][partial-1][0][0] = (pars['magLocal.env'][partial-1][0][1] + pars['magLocal.env'][partial-1][0][2]) / 2
        pars['magLocal.env'][partial-1][1][0] = (pars['magLocal.env'][partial-1][1][1] + pars['magLocal.env'][partial-1][1][2]) / 2
        env1 = linearResample(np.array(pars['magLocal.env'][partial-1][0]), pars_len)
        env2 = linearResample(np.array(pars['magLocal.env'][partial-1][1]), pars_len)
        env1 = env1[start:end]
        env2 = env2[start:end]
        c1 = pars['magLocal.spreadingCenter'][partial-1][0]
        c2 = pars['magLocal.spreadingCenter'][partial-1][1]
        fac1 = pars['magLocal.spreadingFactor'][partial-1][0]
        fac2 = pars['magLocal.spreadingFactor'][partial-1][1]
        ng1 = pars['magLocal.noiseGain'][partial-1][0]
        ng2 = pars['magLocal.noiseGain'][partial-1][1]
        noise_start = round(np.random.rand() * 5e+5)
        phase1 = noise1[noise_start:(noise_start+length_sus)]
        noise_start = round(np.random.rand() * 5e+5)
        phase2 = noise2[noise_start:(noise_start+length_sus)]
        mag_local = np.sin(2*np.pi*c1*t_sus + fac1 * phase1) * env1 * ng1 + np.sin(2*np.pi*c2*t_sus + fac2 * phase2) * env2 * ng2
        
        # most nearing bridge bow position
        if partial % 5 == 0:
            mag_global *= 1.4

        # bow position to timbre
        mag_global *= (1 - (1 - np.clip(np.abs(partial - (1 / bow_pos)), 0 ,1)) * (timbre_gain-1))
        
        # saturation
        if over_freq <= 1000:
            mag_global *= np.power(10 ,(-4 + (saturation_sus * 6)) / 20)
            
        # value
        if over_freq >= 5000:
            mag_global *= np.power(10 ,(-4 + (value_sus * 6)) / 20)

        mag_sus = mag_global + pars['magLocal.gain'][partial-1] * mag_local * noise_fac * mag_fac
        
        # attack AMP
        if partial == 1:
            ma_amp = np.mean(mag_global[:pars['overlapLen']]) / np.mean(mag_attack[-pars['overlapLen']:]) * atk_peak
        mag_attack *= ma_amp
            
        alpha = concat(alpha_attack, alpha_sus, pars['overlapLen'])
        mag = concat(mag_attack, mag_sus, pars['overlapLen'])

        # global pitch
        alpha += np.cumsum(pitch - base_freq) / fs * 2 * np.pi * partial
        
        # release
        decay = release_decay(over_freq)
        mag[-release_len:] *= np.power(decay, np.arange(release_len))
        
        # partial addition
        tone = np.sin(2 * np.pi * over_freq * t + alpha) * mag
        mixtone += tone
        
    sf.write(des, mixtone, fs)
    print('note writen!!')



if __name__ == '__main__':
    instr = 'violin'
    pitch = 67
    name = '08_72'
    input_image_path = Path(f'./data/notation/{name}.png')
    assert input_image_path.exists(), f'{input_image_path} does not exist. please put a notation image here.'
    output_dir = Path(f'data/output/{name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image = cv2.imread(str(input_image_path))
    w, h = image.shape[1], image.shape[0]

    stroke_info_list = notation_to_parameters(input_image_path)
    for stroke_info in stroke_info_list:
        note_construct(stroke_info, w, h, note=pitch, des=f'{output_dir}/{pitch}-{stroke_info.start_x:.0f}-{stroke_info.start_y:.0f}.wav')

    

