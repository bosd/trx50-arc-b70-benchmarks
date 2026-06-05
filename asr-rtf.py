import time, wave, torch
torch.set_num_threads(24)
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.rnnt_bpe_models_prompt import RNNTPromptTranscribeConfig as C
WAV="/home/bosd/asr-bench/test.wav"
with wave.open(WAV) as w: DUR=w.getnframes()/w.getframerate()
m=nemo_asr.models.ASRModel.restore_from("/home/bosd/asr-bench/nemotron-3.5-asr-streaming-0.6b.nemo", map_location="cpu"); m.eval()
cfg=C(batch_size=1, verbose=False, target_lang="en")
def run():
    t=time.time()
    with torch.no_grad(): out=m.transcribe([WAV], override_config=cfg)
    return time.time()-t, out
_,out=run()
o=out[0]; print("transcript:", (o.text if hasattr(o,"text") else str(o))[:160])
ts=sorted(run()[0] for _ in range(5)); med=ts[len(ts)//2]
print("audio %.2fs  infer median %.3fs  RTF=%.4f  xRT=%.1fx"%(DUR,med,med/DUR,DUR/med))
