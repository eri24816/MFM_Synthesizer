from src.parametersGeneration import *

class tableGeneration(parametersGenreration):
    def __init__(self):
        super().__init__()
        self.par_sr = 100
        self.ori_sec = 3
        
    def linearResample(self, y, target_sr):
        l = y.shape[0] - 1
        out = [0] * target_sr
        for i in range(target_sr):
            pos = l / target_sr * i
            floor = int(math.floor(pos))
            cur_diff = (y[floor+1] - y[floor])
            out[i] = y[floor] + cur_diff * (pos - floor)
        return out
        
    def implement(self, instr, genre, note, seq):
        y_ori, _ = librosa.load(f'../original_file/{instr}/{genre}/note{note}-{seq}.wav', sr=self.fs)
        # length = y_ori.shape[0] - self.attack_len - self.release_len + 2 * self.overlap_len
        length = y_ori.shape[0] - self.attack_len + self.overlap_len
        mixtone = np.zeros(shape=(length,) , dtype=np.float32)
        base_freq = 440 * np.power(2, (note-69)/12)
        start = (self.attack_len-self.overlap_len)
        # end = (-self.release_len+self.overlap_len)
        # t = np.arange(length + self.attack_len + self.release_len - 2 * self.overlap_len) / self.fs
        t = np.arange(length + self.attack_len - self.overlap_len) / self.fs
        t_sus = t[start:]

    
        if instr == 'cello':
            self.max_freq = 16000
            
        # initial hilbert for base frequency testing
        base_freq, diff = self.init_base_freq(y_ori, base_freq, order=4)
        max_partial = math.floor(self.max_freq / base_freq)
        print(f'note{note}-{seq}, f0(closed): {base_freq} Hz, max_partial: {max_partial}')
        
        

        # parameters dictionary
        pars = dict()
        pars['sampleRate'] = self.fs
        pars['par_sr'] = self.par_sr
        pars['ori_sec'] = self.ori_sec
        pars['partialAmount'] = max_partial
        pars['pitch'] = base_freq
        pars['coloredCutoff1'] = float()
        pars['coloredCutoff2'] = float()
        pars['alphaGlobal'] = list(list())
        pars['alphaLocal'] = dict()
        pars['alphaLocal']['env'] = list(list())
        pars['alphaLocal']['spreadingCenter'] = list(list())
        pars['alphaLocal']['spreadingFactor'] = list(list())
        pars['alphaLocal']['noiseGain'] = list(list())
        pars['alphaLocal']['gain'] = list()
        pars['totalEnv'] = list()
        pars['magGlobal'] = list(list())
        pars['magRatio'] = list(list())
        pars['magLocal'] = dict()
        pars['magLocal']['env'] = list(list())
        pars['magLocal']['spreadingCenter'] = list(list())
        pars['magLocal']['spreadingFactor'] = list(list())
        pars['magLocal']['noiseGain'] = list(list())
        pars['magLocal']['gain'] = list()
        
        pars['attackLen'] = self.attack_len
        pars['releaseLen'] = self.release_len
        pars['overlapLen'] = self.overlap_len
        pars['alphaAttack'] = list(list())
        pars['alphaRelease'] = list(list())
        pars['magAttack'] = list(list())
        pars['magRelease'] = list(list())
        
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
        pars['totalEnv'] = self.linearResample(envelope_sus, self.par_sr)
        

        # load noise
        noise, _ = librosa.load('./colored_noise.wav', sr=self.fs * (2000 / (base_freq / 2)))
        noise2, _ = librosa.load('./colored_noise.wav', sr=self.fs * (2000 / (base_freq / 8)))
        pars['coloredCutoff1'] = (base_freq / 2)
        pars['coloredCutoff2'] = (base_freq / 8)

        mag_seg = []
        for partial in range(1, max_partial + 1):
            over_freq = base_freq * partial
            signal = self.filter_apply(y_ori, base_freq, over_freq, cutoff_var, self.filter_order, self.filter_times)
            mag, alpha = self.HT(signal, length, over_freq)
            
            # hilbert drop clip (attack & release)
            pars['magAttack'].append(mag[:self.attack_len].tolist())
            # pars['magRelease'].append(mag[-self.release_len:].tolist())
            mag_sus = mag[start:]
            alpha_sus = alpha[start:]
            
            # alpha extract
            z = np.polyfit(t_sus, alpha_sus, 1)
            p = np.poly1d(z)
            alpha -= p(t)
            alpha_attack = alpha[:self.attack_len]
            alpha_release = alpha[-self.release_len:]
            pars['alphaAttack'].append(alpha_attack.tolist())
            # pars['alphaRelease'].append(alpha_release.tolist())
            
            
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
            pars['alphaLocal']['spreadingFactor'].append([spread_fac1, spread_fac2])
            pars['magLocal']['spreadingFactor'].append([spread_fac1, spread_fac2])
            
            ################################
            # alpha local part hilbert analysis
            amp = np.max(np.abs(alpha_local))
            local = alpha_local / amp
            
            center1 = 0
            center2 = base_freq
            pars['alphaLocal']['spreadingCenter'].append([center1, center2])
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
            pars['alphaLocal']['env'].append([self.linearResample(noise_env1, self.par_sr), self.linearResample(noise_env2, self.par_sr)])
            pars['alphaLocal']['noiseGain'].append([recon_local_1st_scale, recon_local_2nd_scale])
            
            recon_local = recon_local_1st + recon_local_2nd
            recon_local_energy = np.mean(np.power(recon_local, 2))
            recon_alpha_local_scale = np.sqrt(alpha_local_energy / recon_local_energy)
            recon_alpha_local = recon_local * recon_alpha_local_scale
            pars['alphaLocal']['gain'].append(recon_alpha_local_scale)
            
            ################################
            # mag local part hilbert analysis
            amp = np.max(np.abs(mag_local))
            local = mag_local / amp
            
            center1 = 0
            center2 = base_freq
            pars['magLocal']['spreadingCenter'].append([center1, center2])
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
            pars['magLocal']['env'].append([self.linearResample(noise_env1, self.par_sr), self.linearResample(noise_env2, self.par_sr)])
            pars['magLocal']['noiseGain'].append([recon_local_1st_scale, recon_local_2nd_scale])

            recon_local = recon_local_1st + recon_local_2nd
            recon_local_energy = np.mean(np.power(recon_local, 2))
            recon_mag_local_scale = np.sqrt(mag_local_energy / recon_local_energy)
            recon_mag_local = recon_local * recon_mag_local_scale
            pars['magLocal']['gain'].append(recon_mag_local_scale) 
            
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
            pars['alphaGlobal'].append(self.linearResample(alpha_global[start:], self.par_sr))
            pars['magGlobal'].append(self.linearResample(mag_global[start:], self.par_sr))
            pars['magRatio'].append(self.linearResample(mag_global[start:] / envelope_sus, self.par_sr))
            
        # sf.write(f'../note_out/{genre}_note{note}_{seq}.wav', mixtone, self.fs)
        with open(f'../table/{instr}/{genre}/note{note}-{seq}.json', 'w+') as jf:
            j = json.dumps(pars, indent=4)
            jf.write(j)