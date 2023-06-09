import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import torch
from wake_words.configs import exp1 as config
from wake_words.src.model import WakeWordDetector

def parse_args():
    return None

def trace(model):
    model.eval()
    x = [torch.rand(1, 16000)]
    traced = torch.jit.trace(model, [x])
    return traced

def main(args):
    model = WakeWordDetector(**config.model)
    ckpt_path = os.path.join(config.trainer.get('ckpt_dir'), 'epoch_0.pt')
    model.load_state_dict(torch.load(ckpt_path))
    
    traced = trace(model)
    traced.save('wake_words/compiled/epoch_0.pt')

if __name__ == '__main__':
    main(parse_args())
