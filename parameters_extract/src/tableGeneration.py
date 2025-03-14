from pathlib import Path
from .parametersGeneration import parametersGeneration
import numpy as np
import math
import librosa
from scipy.signal import iirfilter, freqz, hilbert

class tableGeneration(parametersGeneration):
    def __init__(self):
        super().__init__()
        self.par_sr = 100
        self.ori_sec = 3
        
    # This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
    #             which was released under LGPL. 
    def resample(self, signal, input_fs, output_fs):
    
        scale = output_fs / input_fs
        # calculate new length of sample
        n = round(len(signal) * scale)
    
        # use linear interpolation
        # endpoint keyword means than linspace doesn't go all the way to 1.0
        # If it did, there are some off-by-one errors
        # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
        # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
        # Both are OK, but since resampling will often involve
        # exact ratios (i.e. for 44100 to 22050 or vice versa)
        # using endpoint=False gets less noise in the resampled sound
        resampled_signal = np.interp(
            np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
            np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
            signal,  # known data points
        )
        return resampled_signal
        
    def implement(self, audio_path:Path, table_path:Path, pitch:int, max_n_partial:int=9999999):
        y_ori, _ = librosa.load(audio_path, sr=self.fs)
        y_ori = y_ori.astype(np.float32)
        # length = y_ori.shape[0] - self.attack_len - self.release_len + 2 * self.overlap_len
        length = y_ori.shape[0] - self.attack_len + self.overlap_len
        mixtone = np.zeros(shape=(length,) , dtype=np.float32)
        base_freq = 440 * np.power(2, (pitch-69)/12)
        start = (self.attack_len-self.overlap_len)
        # end = (-self.release_len+self.overlap_len)
        # t = np.arange(length + self.attack_len + self.release_len - 2 * self.overlap_len) / self.fs
        t = np.arange(length + self.attack_len - self.overlap_len) / self.fs
        t_sus = t[start:]
        
        # initial hilbert for base frequency testing
        base_freq, diff = self.init_base_freq(y_ori, base_freq, order=4)
        n_partial = min(max_n_partial, math.floor(self.max_freq / base_freq))
        print(f'note{pitch}, f0(closed): {base_freq} Hz, max_partial: {n_partial}')
        
        

        # parameters dictionary
        pars = dict()
        pars['sampleRate'] = self.fs
        pars['par_sr'] = self.par_sr
        pars['ori_sec'] = self.ori_sec
        pars['partialAmount'] = n_partial
        pars['pitch'] = base_freq
        
        pars['coloredCutoff1'] = float()
        pars['coloredCutoff2'] = float()

        pars['alphaGlobal'] = list(list())

        pars['alphaLocal.env'] = list(list())
        pars['alphaLocal.spreadingCenter'] = list(list())
        pars['alphaLocal.spreadingFactor'] = list(list())
        pars['alphaLocal.noiseGain'] = list(list())
        pars['alphaLocal.gain'] = list()

        pars['totalEnv'] = list()

        pars['magGlobal'] = list(list())
        pars['magRatio'] = list(list())

        pars['magLocal.env'] = list(list())
        pars['magLocal.spreadingCenter'] = list(list())
        pars['magLocal.spreadingFactor'] = list(list())
        pars['magLocal.noiseGain'] = list(list())
        pars['magLocal.gain'] = list()
        
        pars['attackLen'] = self.attack_len
        pars['releaseLen'] = self.release_len
        pars['overlapLen'] = self.overlap_len
        pars['alphaAttack'] = list(list())
        pars['alphaRelease'] = list(list())
        pars['magAttack'] = list(list())
        pars['magRelease'] = list(list())
        pars['attackWave'] = list(list())
        
        # band pass filter width test
        b, a = iirfilter(self.filter_order, Wn=(base_freq * 0.5, base_freq * 1.5), fs=self.fs, ftype="butter", btype="band")
        w, h = freqz(b, a, worN=22050)
        for i in range(round(base_freq), 22050):
            if abs(h[i]) <= np.power(1 / 2, 1 / self.filter_times):
                cutoff_var = i - round(base_freq * 1.5)
                break
            
        # total envelope extract
        mag, phase = self.HT(y_ori, length, base_freq)
        split_b, split_a = iirfilter(self.split_filter_order, Wn=self.split_filter_cutoff, fs=self.fs, ftype="butter", btype="lowpass")
        envelope = self.timesfilt(split_b, split_a, mag, self.split_filter_times)
        envelope_sus = envelope[start:]
        pars['totalEnv'] = self.resample(envelope_sus, self.fs, self.par_sr)
        

        # load noise
        noise, _ = librosa.load('parameters_extract/colored_noise.wav', sr=self.fs * (2000 / (base_freq / 2)))
        noise2, _ = librosa.load('parameters_extract/colored_noise.wav', sr=self.fs * (2000 / (base_freq / 8)))
        pars['coloredCutoff1'] = (base_freq / 2)
        pars['coloredCutoff2'] = (base_freq / 8)


        pars['attackWave'].append(y_ori[:self.attack_len])

        mag_seg = []
        for partial in range(1, n_partial + 1):
            over_freq = base_freq * partial


            signal = self.filter_apply(y_ori, base_freq, over_freq, cutoff_var, self.filter_order, self.filter_times)
            mag, alpha = self.HT(signal, length, over_freq)
            
            '''
            Attack
            '''


            # hilbert drop clip (attack & release)
            pars['magAttack'].append(mag[:self.attack_len].tolist())
            pars['magRelease'].append(mag[-self.release_len:].tolist())
            mag_sus = mag[start:]
            alpha_sus = alpha[start:]
            
            # alpha extract
            z = np.polyfit(t_sus, alpha_sus, 1)
            p = np.poly1d(z)
            alpha -= p(t)
            alpha_attack = alpha[:self.attack_len]
            alpha_release = alpha[-self.release_len:]
            pars['alphaAttack'].append(alpha_attack.tolist())
            pars['alphaRelease'].append(alpha_release.tolist())
            
            '''
            Sustain
            '''
            
            alpha_global = self.timesfilt(split_b, split_a, alpha, self.split_filter_times)
            alpha_local = alpha - alpha_global
            
            # mag extract
            mag_global = self.timesfilt(split_b, split_a, mag, self.split_filter_times)
            mag_local = mag - mag_global
            
            # local energy
            alpha_local_energy = np.mean(np.power(alpha_local[start:], 2))
            mag_local_energy = np.mean(np.power(mag_local[start:], 2))
            
            spread_fac1 = 0.01
            spread_fac2 = 0.3
            pars['alphaLocal.spreadingFactor'].append([spread_fac1, spread_fac2])
            pars['magLocal.spreadingFactor'].append([spread_fac1, spread_fac2])
            
            ################################
            # alpha local part hilbert analysis
            amp = np.max(np.abs(alpha_local))
            local = alpha_local / amp
            
            center1 = 0
            center2 = base_freq
            pars['alphaLocal.spreadingCenter'].append([center1, center2])
            b3, a3 = iirfilter(self.filter_order, Wn=base_freq, fs=self.fs, ftype="butter", btype="lowpass")
            local_1st = self.timesfilt(b3, a3, local, self.filter_times / 2)
            local_2nd = local - local_1st
            local_1st_energy = np.mean(np.power(local_1st, 2))
            local_2nd_energy = np.mean(np.power(local_2nd, 2))
            
            analytic_signal = hilbert(local_1st)
            instantaneous_magnitude = np.abs(analytic_signal)
            noise_env1 = self.timesfilt(split_b, split_a, instantaneous_magnitude[start:], self.split_filter_times)
            noise_start = round(np.random.rand() * 5e+5)
            phase = noise[noise_start:(noise_start+length)]
            recon_local_1st = np.sin(2*np.pi*center1*t_sus + spread_fac1 * phase) * noise_env1
            recon_local_1st_energy = np.mean(np.power(recon_local_1st, 2))
            recon_local_1st_scale = np.sqrt(local_1st_energy / recon_local_1st_energy)
            recon_local_1st = recon_local_1st * recon_local_1st_scale
            
            analytic_signal = hilbert(local_2nd)
            instantaneous_magnitude = np.abs(analytic_signal)
            noise_env2 = self.timesfilt(split_b, split_a, instantaneous_magnitude[start:], self.split_filter_times)
            noise_start = round(np.random.rand() * 5e+5)
            phase = noise2[noise_start:(noise_start+length)]
            recon_local_2nd = np.sin(2*np.pi*center2*t_sus + spread_fac2 * phase) * noise_env2
            recon_local_2nd_energy = np.mean(np.power(recon_local_2nd, 2))
            recon_local_2nd_scale = np.sqrt(local_2nd_energy / recon_local_2nd_energy)
            recon_local_2nd = recon_local_2nd * recon_local_2nd_scale
            pars['alphaLocal.env'].append([self.resample(noise_env1, self.fs, self.par_sr), self.resample(noise_env2, self.fs, self.par_sr)])
            pars['alphaLocal.noiseGain'].append([recon_local_1st_scale, recon_local_2nd_scale])
            
            recon_local = recon_local_1st + recon_local_2nd
            recon_local_energy = np.mean(np.power(recon_local, 2))
            recon_alpha_local_scale = np.sqrt(alpha_local_energy / recon_local_energy)
            recon_alpha_local = recon_local * recon_alpha_local_scale
            pars['alphaLocal.gain'].append(recon_alpha_local_scale)
            
            ################################
            # mag local part hilbert analysis
            amp = np.max(np.abs(mag_local))
            local = mag_local / amp
            
            center1 = 0
            center2 = base_freq
            pars['magLocal.spreadingCenter'].append([center1, center2])
            b3, a3 = iirfilter(self.filter_order, Wn=base_freq, fs=self.fs, ftype="butter", btype="lowpass")
            local_1st = self.timesfilt(b3, a3, local, self.filter_times / 2)
            local_2nd = local - local_1st
            local_1st_energy = np.mean(np.power(local_1st, 2))
            local_2nd_energy = np.mean(np.power(local_2nd, 2))
    
            analytic_signal = hilbert(local_1st)
            instantaneous_magnitude = np.abs(analytic_signal)
            noise_env1 = self.timesfilt(split_b, split_a, instantaneous_magnitude[start:], self.split_filter_times)
            noise_start = round(np.random.rand() * 5e+5)
            phase = noise[noise_start:(noise_start+length)]
            recon_local_1st = np.sin(2*np.pi*center1*t_sus + spread_fac1 * phase) * noise_env1
            recon_local_1st_energy = np.mean(np.power(recon_local_1st, 2))
            recon_local_1st_scale = np.sqrt(local_1st_energy / recon_local_1st_energy)
            recon_local_1st = recon_local_1st * recon_local_1st_scale
            
            analytic_signal = hilbert(local_2nd)
            instantaneous_magnitude = np.abs(analytic_signal)
            noise_env2 = self.timesfilt(split_b, split_a, instantaneous_magnitude[start:], self.split_filter_times)
            noise_start = round(np.random.rand() * 5e+5)
            phase = noise2[noise_start:(noise_start+length)]
            recon_local_2nd = np.sin(2*np.pi*center2*t_sus + spread_fac2 * phase) * noise_env2
            recon_local_2nd_energy = np.mean(np.power(recon_local_2nd, 2))
            recon_local_2nd_scale = np.sqrt(local_2nd_energy / recon_local_2nd_energy)
            recon_local_2nd = recon_local_2nd * recon_local_2nd_scale
            pars['magLocal.env'].append([self.resample(noise_env1, self.fs, self.par_sr), self.resample(noise_env2, self.fs, self.par_sr)])
            pars['magLocal.noiseGain'].append([recon_local_1st_scale, recon_local_2nd_scale])

            recon_local = recon_local_1st + recon_local_2nd
            recon_local_energy = np.mean(np.power(recon_local, 2))
            recon_mag_local_scale = np.sqrt(mag_local_energy / recon_local_energy)
            recon_mag_local = recon_local * recon_mag_local_scale
            pars['magLocal.gain'].append(recon_mag_local_scale) 
            
            # noisy_fac = 1.0
            # recon_alpha = alpha_global + recon_alpha_local * noisy_fac
            # recon_mag = mag_global + recon_mag_local * noisy_fac
            
            # build mixtone
            # recon_mag = mag
            # recon_alpha = alpha
            # t = np.arange(recon_mag.shape[0]) / self.fs
            # tone = np.sin(2 * np.pi * over_freq * t + recon_alpha) * recon_mag
            # mixtone += tone
            
            # pars write
            pars['alphaGlobal'].append(self.resample(alpha_global[start:], self.fs, self.par_sr))
            pars['magGlobal'].append(self.resample(mag_global[start:], self.fs, self.par_sr))
            pars['magRatio'].append(self.resample(mag_global[start:] / envelope_sus, self.fs, self.par_sr))

            # check nan in all pars
            for key, value in pars.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[-1], np.ndarray):
                    if np.isnan(value[-1]).any():
                        print(f'{key} has nan at pitch: {pitch}, partial: {partial}, {over_freq} Hz')
        
        # convert all float64 to float32
        for key, value in pars.items():
            value = np.array(value)
            if value.dtype == np.float64:
                pars[key] = value.astype(np.float32)
            
        # sf.write(f'../note_out/{genre}_note{note}_{seq}.wav', mixtone, self.fs)
        with open(table_path, 'wb') as f:
            np.savez(f, **pars)
            