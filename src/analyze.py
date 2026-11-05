import numpy as np, subprocess, json
sr = 22050
raw = subprocess.run(['ffmpeg','-v','quiet','-i','assets/song.mp3','-ac','1','-ar',str(sr),'-f','f32le','-'],capture_output=True).stdout
y = np.frombuffer(raw, np.float32)
hop, n = 256, 1024
frames = np.lib.stride_tricks.sliding_window_view(y, n)[::hop] * np.hanning(n)
S = np.abs(np.fft.rfft(frames, axis=1))
logS = np.log1p(S*10)
flux = np.maximum(0, np.diff(logS, axis=0)).sum(1); flux = np.r_[0, flux]
t = np.arange(len(flux))*hop/sr
freqs = np.fft.rfftfreq(n, 1/sr)
low = logS[:, freqs < 150].mean(1)
voc = logS[:, (freqs > 300) & (freqs < 3000)].mean(1)
rms = np.sqrt((frames**2).mean(1))
# tempo via autocorrelation of flux
f = flux - flux.mean()
ac = np.correlate(f, f, 'full')[len(f)-1:]
lags = np.arange(len(ac))*hop/sr
mask = (lags > 60/200) & (lags < 60/70)
best = lags[mask][np.argmax(ac[mask])]
print('tempo', 60/best)
# peaks
thr = np.convolve(flux, np.ones(15)/15, 'same')*1.4 + flux.std()*0.3
pk = [i for i in range(1, len(flux)-1) if flux[i] > thr[i] and flux[i] >= flux[i-1] and flux[i] >= flux[i+1]]
on = []
for i in pk:
    if not on or t[i]-on[-1] > 0.12: on.append(round(float(t[i]),3))
print('onsets', len(on), on)
for s in np.arange(0, 17.5, 0.25):
    i = (t >= s) & (t < s+0.25)
    print(f"{s:5.2f} rms {rms[i].mean():.3f} low {low[i].mean():.2f} voc {voc[i].mean():.2f} " + '#'*int(rms[i].mean()*150))
json.dump({'tempo': 60/best, 'onsets': on}, open('src/analysis.json','w'))
