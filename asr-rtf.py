import time, os, sys, contextlib, wave
os.environ["OMP_NUM_THREADS"]=os.environ.get("OMP_NUM_THREADS","24")
import torch; torch.set_num_threads(24)
import nemo.collections.asr as nemo_asr
WAV="/home/bosd/asr-bench/test.wav"
with wave.open(WAV) as w: DUR=w.getnframes()/w.getframerate()
t0=time.time()
m=nemo_asr.models.ASRModel.restore_from("/home/bosd/asr-bench/nemotron-speech-streaming-en-0.6b.nemo", map_location="cpu")
m.eval()
print(f"load: {time.time()-t0:.1f}s  audio: {DUR:.2f}s")
def run():
    t=time.time()
    with torch.no_grad():
        out=m.transcribe([WAV], batch_size=1, verbose=False)
    return time.time()-t, out
# warmup
try:
    wt,out=run()
except Exception as e:
    print("transcribe error:", repr(e)[:300]); sys.exit(1)
txt=out[0].text if hasattr(out[0],"text") else str(out[0])
print("transcript:", txt[:200])
# timed
times=[run()[0] for _ in range(5)]
times.sort(); med=times[len(times)//2]
print(f"infer median: {med:.3f}s over 5 runs  (min {min(times):.3f})")
print(f"RTF = {med/DUR:.4f}   xRealtime = {DUR/med:.1f}x")
