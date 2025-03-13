import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, freqz, filtfilt, hilbert, sosfiltfilt
import librosa
import soundfile as sf
import math
import os
import json


# vib_phase = [5.182555016327718, 2.0607366837127103, 1.7788474272636459, 3.2976835702205434, 4.94273878322926, 1.6467381339427416, 5.0588603276769355, 5.361786095801302, 1.8024083203399859, 2.410784237989757, 2.750734266662658, 1.6391649897396323, 5.421529788959163, 1.5819456779828074, 5.003323936854134, 4.827458699248599, 4.866165880731157, 1.9462980601990603, 1.5625920872415282, 1.138496011867414, 1.65010397581079, 1.7822132691316948, 5.262493760693871, 2.57402756859011, 4.134095274430602, 3.0873184534675104, 4.309960512036137, 5.5814072776914685, 0.0, 1.990054004483691, 3.275805598078228, 2.8323559319628346, 3.4331587054094967, 4.84765375045689, 2.6901491130377844, 3.449146454282727, 4.0053518229777465, 0.46280325685667234, 4.558191349804717, 4.951994848366394, 5.015104383392304, 0.49477875460313336, 3.895120501799157, 4.1685951535781, 2.8946240065217324, 4.503496419448928, 4.5817522428810555, 5.453505286705624, 3.7655355898792884, 0.8153751925347554]
vib_phase = [5.536809872939826, 2.849185141303077, 2.828148629627774, 0.6765342154777538, 5.38450552841063, 1.8983348135793685, 5.049604262539801, 6.273929242042453, 2.780185383008083, 3.579572826669608, 0.5402176198217884, 2.1162730745355107, 0.09929233510743152, 1.5844700593838439, 1.7401402457810882, 1.6829209340242632, 1.4616168312000724, 2.2795164051358645, 2.624515196610838, 1.1805690352180205, 2.6623809176263844, 2.430137828731036, 0.30124284719034305, 3.7108406595235, 0.0016829209340242632, 3.084794072066474, 1.4296413334536116, 5.580565817224456, 5.291103416572283, 0.9575820114598058, 1.018167165084679, 4.384009033133205, 0.4745837033948421, 1.7283597992429183, 0.6151076013858682, 1.036679295358946, 1.2950076587316706, 1.7796888877306583, 1.3076295657368524, 0.7573144203109184, 5.010055620590231, 0.4956202150701455, 4.449642949560152, 5.509883137995438, 2.9005142297908173, 1.0526670442321766, 1.6492625153437777, 1.931993232259854, 5.138799072043088, 0.8010703645955491]
# vib_amp = [0.004387044282123212, 0.008028885758092675, 0.0024723642804005963, 0.003351917008762816, 0.0065980919368036055, 0.0040756521015001546, 0.0019150360914909912, 0.0006805338980789058, 0.0013240639511691537, 0.0002892146027922755, 6.574529134912167e-05, 0.0005246572510019978, 0.0003824681844240474, 0.0009143299172050327, 0.00015652795551135775, 0.00024220135307101388, 0.0002204727952057935, 8.180619937182346e-05, 9.987636027550228e-05, 5.579105877045497e-05, 5.837040945413734e-05, 5.8536217467200426e-05, 1.8365839854847652e-05, 3.1868117240515284e-05, 7.5885793334525426e-06, 1.808304389092854e-05, 2.2985958502444482e-05, 1.7165867142425264e-05, 1.6146095162022526e-05, 1.5403935535413787e-05, 2.0403820773289243e-05, 1.4972721759351218e-05, 1.2479397918173875e-05, 8.488453338350872e-06, 7.176441318455684e-06, 5.908493312332547e-06, 4.9579962648387e-06, 4.323055715499605e-06, 4.972635163415998e-06, 4.2133995474395384e-06, 4.809327867667732e-06, 3.5996672030791224e-06, 1.8864605425815022e-06, 1.6460476605084939e-06, 1.8132999945290033e-06, 1.9876378825124272e-06, 2.608153548075658e-06, 1.88551230011061e-06, 8.951831101108585e-07, 8.08557365289395e-07]
vib_amp = [0.14357095774419332, 0.33635365095684117, 0.25391482481587757, 0.05075399254764121, 0.3930651351648941, 0.36554165678333406, 0.22940795822238047, 0.2530248819095757, 0.6606079930461508, 0.2602258452265681, 0.08604641553385403, 0.5770706306058275, 0.2629444325911669, 0.5156180427162119, 0.15290505902344448, 0.5549476331157179, 0.2653223650354921, 0.3666558256016987, 0.5467630142249739, 0.2482521946260533, 0.2304233258784584, 0.428719925169106, 0.23261003048772377, 0.15640063695322717, 0.11359505673365823, 0.2275863265737768, 0.13665261055951763, 0.12528825595253473, 0.20302297261717295, 0.1601859462826407, 0.28574663084587243, 0.10306470546579707, 0.16114502041667744, 0.10052206098827093, 0.05243605528626826, 0.0913077872686589, 0.1311037322678462, 0.07746522436760915, 0.23096392136116584, 0.3742851037714845, 0.33640400961888883, 0.21873872920014595, 0.1886627342798931, 0.28548572060658534, 0.061401591160784155, 0.09557714451423308, 0.3027531086955034, 0.32617508941824996, 0.16107082832346206, 0.14403599690235006]
vib_freq = 6


class parametersGeneration:
    def __init__(self):
        self.fs = 44100
        self.max_freq = 20000
        self.fft_size = 2048
        self.hop_size = int(self.fft_size / 2)
        self.bins_width = self.fs / self.fft_size
        self.filter_order = 4
        self.filter_times = 8
        self.split_filter_cutoff = 20
        self.split_filter_order = 4
        self.split_filter_times = 8
        
        self.hilbert_drop_len = 10000
        self.attack_len = round(self.fs * 0.12)
        self.release_len = round(self.fs * 0.02)
        self.overlap_len = round(self.fs * 0.02)
        self.hilbert_window = 2048
        self.hilbert_hop = int(self.hilbert_window / 2)
        self.ma_win = 10000
        
        self.plot = 0
        self.write = 1
        self.vibration = 0

    def timesfilt(self, b, a, signal, times):
        output = signal
        for _ in range(round(times / 2)):
            output = filtfilt(b, a, output)
        return output
    
    def timessosfiltfilt(self, sos, signal, times):
        output = signal
        for _ in range(round(times / 2)):
            output = sosfiltfilt(sos, output)
        return output

    def filter_apply(self, signal, base_freq, over_freq, cutoff_var, order, times):
        lcf = (over_freq)-(base_freq/2 - cutoff_var)
        hcf = (over_freq)+(base_freq/2 - cutoff_var)
        if base_freq == over_freq:
            sos = iirfilter(order, Wn=hcf, fs=self.fs, ftype="butter", btype="lowpass", output='sos')
        else:
            sos = iirfilter(order, Wn=(lcf, hcf), fs=self.fs, ftype="butter", btype="band", output='sos')    
        output = self.timessosfiltfilt(sos, signal, times)
        return output

    def init_base_freq(self, y_clip, base_freq, order):
        for _ in range(3):
            sos = iirfilter(order, Wn=(base_freq * 1.75, base_freq * 2.25), fs=self.fs, ftype="butter", btype="band", output='sos')
            # b, a = iirfilter(order, Wn=(base_freq * 0.75, base_freq * 1.25), fs=fs, ftype="cheby1", btype="band", rp=0.4)
            signal = y_clip
            for _ in range(4):
                signal = sosfiltfilt(sos, signal)
            analytic_signal :np.ndarray = hilbert(signal) # type: ignore
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * self.fs)
            new_base_freq = np.mean(instantaneous_frequency[8000:40000]) / 2 # use 0.2s to 4s for freq estimation
            diff = new_base_freq - base_freq
        return new_base_freq, diff
    
    def HT(self, signal, length, over_freq):
        t = np.arange(length + self.attack_len + self.release_len - 2 * self.overlap_len, dtype=np.float32) / self.fs
        analytic_signal = hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal)).astype(np.float32)
        instantaneous_magnitude = np.abs(analytic_signal).astype(np.float32)
        origin_phase = t * (2 * np.pi * over_freq) - 0.5 * np.pi
        phase_diff = instantaneous_phase - origin_phase
        return instantaneous_magnitude, phase_diff

    def implement(self, instr, genre, note, seq):
        if self.write and not os.path.exists(f'../parameters_output/{genre}/note{note}-{seq}/'):
            os.makedirs(f'../parameters_output/{genre}/note{note}-{seq}/alpha')
            os.makedirs(f'../parameters_output/{genre}/note{note}-{seq}/alphaGlobal')
            os.makedirs(f'../parameters_output/{genre}/note{note}-{seq}/pitchVar')
            os.makedirs(f'../parameters_output/{genre}/note{note}-{seq}/alphaLocal')
            os.makedirs(f'../parameters_output/{genre}/note{note}-{seq}/mag')
            os.makedirs(f'../parameters_output/{genre}/note{note}-{seq}/magGlobal')
            os.makedirs(f'../parameters_output/{genre}/note{note}-{seq}/magLocal')
            
            
        y_ori, _ = librosa.load(f'../original_file/{genre}/note{note}-{seq}.wav', sr=self.fs)
        length = y_ori.shape[0] - 2*self.hilbert_drop_len
        mixtone = np.zeros(shape=(length,) , dtype=np.float32)
        base_freq = 440 * np.power(2, (note-69)/12)

        # initial hilbert for base frequency testing
        base_freq, diff = self.init_base_freq(y_ori, base_freq, order=4)
        max_partial = math.floor(self.max_freq / base_freq)
        print(f'note{note}-{seq}, f0(closed): {base_freq} Hz, max_partial: {max_partial}')

        # parameters dictionary
        pars = dict()
        pars['sampleRate'] = self.fs
        pars['sampleLen'] = length
        pars['partialAmount'] = max_partial
        pars['pitch'] = base_freq
        pars['scale'] = dict()
        pars['scale']['alpha'] = [0] * max_partial
        pars['scale']['alphaLocal'] = [0] * max_partial
        pars['scale']['alphaGlobal'] = [0] * max_partial
        pars['scale']['mag'] = [0] * max_partial
        pars['scale']['magLocal'] = [0] * max_partial
        pars['scale']['magGlobal'] = [0] * max_partial
        pars['scale']['pitchVar'] = [0] * max_partial
        
        # pars['alpha'] = np.zeros(shape=(max_partial, length)).tolist()
        # pars['alphaLocal'] = np.zeros(shape=(max_partial, length)).tolist()
        # pars['alphaGlobal'] = np.zeros(shape=(max_partial, length)).tolist()
        # pars['mag'] = np.zeros(shape=(max_partial, length)).tolist()
        # pars['magLocal'] = np.zeros(shape=(max_partial, length)).tolist()
        # pars['magGlobal'] = np.zeros(shape=(max_partial, length)).tolist()
        # pars['pitchVar'] = np.zeros(shape=(max_partial, length)).tolist()
        
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
        envelope = self.timesfilt(b, a, mag[self.hilbert_drop_len:-self.hilbert_drop_len], self.split_filter_times)

        # load noise
        noise, _ = librosa.load('./colored_noise.wav', sr=self.fs * (2000 / (base_freq / 2)))
        noise2, _ = librosa.load('./colored_noise.wav', sr=self.fs * (2000 / (base_freq / 8)))

        mag_seg = []

        stretch_fac = 1.0
        envelope = librosa.resample(envelope, orig_sr=self.fs, target_sr=round(self.fs * stretch_fac))
        mixtone = np.zeros(shape=(round(mixtone.shape[0] * stretch_fac), ))
        for partial in range(1, max_partial + 1):
            over_freq = base_freq * partial
            signal = self.filter_apply(y_ori, base_freq, over_freq, cutoff_var, self.filter_order, self.filter_times)
            partial_scale = np.max(np.abs(signal))
            mag, phase = self.HT(signal / partial_scale, length, over_freq)
            
            # hilbert drop clip
            t = np.arange(length + 2*self.hilbert_drop_len) / self.fs
            t = t[self.hilbert_drop_len:-self.hilbert_drop_len]
            mag = mag[self.hilbert_drop_len:-self.hilbert_drop_len]
            phase = phase[self.hilbert_drop_len:-self.hilbert_drop_len]
            
            # alpha extract
            z = np.polyfit(t, phase, 1)
            p = np.poly1d(z)
            pitch_shift = p(t)
            alpha = phase - pitch_shift
            
            alpha_global = self.timesfilt(split_b, split_a, alpha, self.split_filter_times)
            alpha_local = alpha - alpha_global
            freq_var = (np.diff(alpha_global) / (2.0*np.pi) * self.fs)
            freq_var = np.append([0], freq_var)
            
            # mag extract
            mag *= partial_scale
            mag_global = self.timesfilt(split_b, split_a, mag, self.split_filter_times)
            mag_local = mag - mag_global
            
            partial_ratio = mag_global / envelope
            if partial == 1:
                base_ratio = partial_ratio
            
            # local energy
            alpha_local_energy = np.mean(np.power(alpha_local, 2))
            mag_local_energy = np.mean(np.power(mag_local, 2))
            
            
            # spread_fac = 0.01
            # # alpha local part hilbert analysis
            # amp = np.max(np.abs(mag_local))
            # local = mag_local / amp
            
            # center = base_freq
            # b3, a3 = iirfilter(self.filter_order, Wn=base_freq, fs=self.fs, ftype="butter", btype="lowpass")
            # local_1st = self.timesfilt(b3, a3, local, self.filter_times / 2)
            # local_2nd = local - local_1st
            # local_1st_energy = np.mean(np.power(local_1st, 2))
            # local_2nd_energy = np.mean(np.power(local_2nd, 2))
            
            # analytic_signal = hilbert(local_1st)
            # instantaneous_magnitude = np.abs(analytic_signal)
            # noise_env_global = self.timesfilt(b, a, instantaneous_magnitude, self.split_filter_times)
            # noise_start = round(np.random.rand() * 5e+5)
            # phase = noise[noise_start:(noise_start+length)]
            # recon_local_1st = np.sin(spread_fac * phase) * noise_env_global
            # recon_local_1st_energy = np.mean(np.power(recon_local_1st, 2))
            # recon_local_1st = recon_local_1st * np.sqrt(local_1st_energy / recon_local_1st_energy)
            
            # analytic_signal = hilbert(local_2nd)
            # instantaneous_magnitude = np.abs(analytic_signal)
            # noise_env_global = self.timesfilt(b, a, instantaneous_magnitude, self.split_filter_times)
            # noise_start = round(np.random.rand() * 5e+5)
            # phase = noise2[noise_start:(noise_start+length)]
            # recon_local_2nd = np.sin(2*np.pi*center*t + spread_fac * 30 * phase) * noise_env_global
            # recon_local_2nd_energy = np.mean(np.power(recon_local_2nd, 2))
            # recon_local_2nd = recon_local_2nd * np.sqrt(local_2nd_energy / recon_local_2nd_energy)

            # recon_local = recon_local_1st + recon_local_2nd
            # recon_local_energy = np.mean(np.power(recon_local, 2))
            # recon_mag_local = recon_local * np.sqrt(mag_local_energy / recon_local_energy)
            
            # # mag local part hilbert analysis
            # amp = np.max(np.abs(alpha_local))
            # local = alpha_local / amp
            
            # center = base_freq
            # b3, a3 = iirfilter(self.filter_order, Wn=base_freq, fs=self.fs, ftype="butter", btype="lowpass")
            # local_1st = self.timesfilt(b3, a3, local, self.filter_times / 2)
            # local_2nd = local - local_1st
            # local_1st_energy = np.mean(np.power(local_1st, 2))
            # local_2nd_energy = np.mean(np.power(local_2nd, 2))
    
            # analytic_signal = hilbert(local_1st)
            # instantaneous_magnitude = np.abs(analytic_signal)
            # noise_env_global = self.timesfilt(b, a, instantaneous_magnitude, self.split_filter_times)
            # noise_start = round(np.random.rand() * 5e+5)
            # phase = noise[noise_start:(noise_start+length)]
            # recon_local_1st = np.sin(spread_fac * phase) * noise_env_global
            # recon_local_1st_energy = np.mean(np.power(recon_local_1st, 2))
            # recon_local_1st = recon_local_1st * np.sqrt(local_1st_energy / recon_local_1st_energy)
            
            # analytic_signal = hilbert(local_2nd)
            # instantaneous_magnitude = np.abs(analytic_signal)
            # noise_env_global = self.timesfilt(b, a, instantaneous_magnitude, self.split_filter_times)
            # noise_start = round(np.random.rand() * 5e+5)
            # phase = noise2[noise_start:(noise_start+length)]
            # recon_local_2nd = np.sin(2*np.pi*center*t + spread_fac * 30 * phase) * noise_env_global
            # recon_local_2nd_energy = np.mean(np.power(recon_local_2nd, 2))
            # recon_local_2nd = recon_local_2nd * np.sqrt(local_2nd_energy / recon_local_2nd_energy)

            # recon_local = recon_local_1st + recon_local_2nd
            # recon_local_energy = np.mean(np.power(recon_local, 2))
            # recon_alpha_local = recon_local * np.sqrt(alpha_local_energy / recon_local_energy)
            
            
            # noisy_fac = 1.0
            # recon_alpha = alpha_global + recon_alpha_local * noisy_fac
            # recon_mag = mag_global + recon_mag_local * noisy_fac
            
            # adding vibration
            # if vibration:
            #     alpha += np.sin(2*np.pi*vib_freq*t) * over_freq / 400
            #     mag += np.sin(2*np.pi*vib_freq*t + vib_phase[partial-1]) * vib_amp[partial-1] * mag_global
            
            # build mixtone
            recon_mag = mag
            recon_alpha = alpha
            t = np.arange(recon_mag.shape[0]) / self.fs
            tone = np.sin(2 * np.pi * over_freq * t + recon_alpha) * recon_mag
            mixtone += tone
            
            # pars write
            # pars['alpha'][partial-1] = alpha.tolist()
            # pars['alphaLocal'][partial-1] = alpha_local.tolist()
            # pars['alphaGlobal'][partial-1] = alpha_global.tolist()
            # pars['mag'][partial-1] = mag.tolist()
            # pars['magLocal'][partial-1] = mag_local.tolist()
            # pars['magGlobal'][partial-1] = mag_global.tolist()
            # pars['pitchVar'][partial-1] = freq_var.tolist()
            
            
            
            # parameters curve predict plots
            if partial <= 3 and self.plot == True:
                fig, ax = plt.subplots(2,4)
                fig.set_figheight(8)
                fig.set_figwidth(12)
                
                # local part
                ax[0, 0].set_title(f' partial{partial}, mag_local')
                ax[0, 0].plot(t, mag_local)
                ax[0, 0].set_xlabel('sec')
                ax[0, 1].set_title('alpha_local')
                ax[0, 1].plot(t, alpha_local)
                ax[0, 1].set_xlabel('sec')
                
                # local zoom-in
                start = 5000
                field = 1000
                ax[0, 2].set_title('mag_local zoom-in')
                ax[0, 2].plot(t[start:(start+field)], mag_local[start:(start+field)])
                ax[0, 2].set_xlabel('sec')
                ax[0, 3].set_title('alpha_local zoom-in')
                ax[0, 3].plot(t[start:(start+field)], alpha_local[start:(start+field)])
                ax[0, 3].set_xlabel('sec')
                
                # global part
                s = 50113
                e = 50113 + 3984 * 2
                ax[1, 0].set_title('mag_global')
                ax[1, 0].plot(t[20000:50000], mag_global[20000:50000], label='total', linewidth=1)
                # ax[1, 0].plot(t[s:], mag_global[s:], label='global')
                ax[1, 0].legend()
                ax[1, 0].set_xlabel('sec')
                # ax[1, 1].set_title('ratio')
                # ax[1, 1].plot(t, partial_ratio, label='cur_partial')
                # ax[1, 1].plot(t, base_ratio, label='first_partial')
                # ax[1, 1].set_xlabel('sec')
                # ax[1, 1].legend()
                ax[1, 1].set_title('ratio')
                ax[1, 1].plot(partial_ratio, base_ratio)
                
                ax[1, 2].set_title('alpha_global')
                ax[1, 2].plot(t, alpha_global)
                ax[1, 2].set_xlabel('sec')
                ax[1, 3].set_title('pitch variance(Hz)')
                ax[1, 3].plot(t[1:], freq_var)
                ax[1, 3].set_xlabel('sec')
                plt.show()
                
                
            if self.write == True:
                mag_scale = np.max(np.abs(mag))
                mag_global_scale = np.max(np.abs(mag_global))
                mag_local_scale = np.max(np.abs(mag_local))
                alpha_scale = np.max(np.abs(alpha))
                alpha_global_scale = np.max(np.abs(alpha_global))
                alpha_local_scale = np.max(np.abs(alpha_local))
                freq_var_scale = np.max(np.abs(freq_var))
                
                pars['scale']['mag'][partial-1] = mag_scale
                pars['scale']['magGlobal'][partial-1] = mag_global_scale
                pars['scale']['magLocal'][partial-1] = mag_local_scale
                pars['scale']['alpha'][partial-1] = alpha_scale 
                pars['scale']['alphaGlobal'][partial-1] = alpha_global_scale
                pars['scale']['alphaLocal'][partial-1] = alpha_local_scale
                pars['scale']['pitchVar'][partial-1] = freq_var_scale
                
                sf.write(f'../parameters_output/{genre}/note{note}-{seq}/mag/partial{partial}.wav', mag / mag_scale, self.fs)
                sf.write(f'../parameters_output/{genre}/note{note}-{seq}/magGlobal/partial{partial}.wav', mag_global / mag_global_scale, self.fs)
                sf.write(f'../parameters_output/{genre}/note{note}-{seq}/magLocal/partial{partial}.wav', mag_local / mag_local_scale, self.fs)
                sf.write(f'../parameters_output/{genre}/note{note}-{seq}/alpha/partial{partial}.wav', alpha / alpha_scale, self.fs)
                sf.write(f'../parameters_output/{genre}/note{note}-{seq}/alphaGlobal/partial{partial}.wav', alpha_global / alpha_global_scale, self.fs)
                sf.write(f'../parameters_output/{genre}/note{note}-{seq}/alphaLocal/partial{partial}.wav', alpha_local / alpha_local_scale, self.fs)
                sf.write(f'../parameters_output/{genre}/note{note}-{seq}/pitchVar/partial{partial}.wav', freq_var / freq_var_scale, self.fs)
            
        sf.write(f'../note_out/{genre}_note{note}_{seq}.wav', mixtone, self.fs)
        with open(f'../parameters_output/{genre}/note{note}-{seq}/attribute.json', 'w+') as jf:
            j = json.dumps(pars, indent=4)
            jf.write(j)
            
    