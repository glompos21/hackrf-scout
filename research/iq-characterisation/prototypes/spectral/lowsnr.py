"""Where does each cue stop working?  SNR below 0 dB, seeds 100-109, frozen thresholds."""
import sys
sys.path.insert(0, '.')
import harness
CASES = ['cw', 'am-tone', 'nfm-voice', 'ook-pwm', 'fsk2-rect', 'qpsk', 'lora-sf7', 'gfsk4', 'ofdm2', 'ofdm10', 'hopper2', 'hopper10', 'multi-cw+nfm']
SNR = [-15, -12, -9, -6, -3]
tasks = [(c, s, sd, cond) for c in CASES for s in SNR for sd in range(100, 110) for cond in ('clean', 'imp')]
if __name__ == '__main__':
    harness.run_tasks(tasks, 'results/lowsnr.jsonl')
