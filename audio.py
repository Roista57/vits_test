import matplotlib.pyplot as plt
from IPython.display import Audio

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def create_audio(global_step):
    hps = utils.get_hparams_from_file("/content/vits_test/configs/ljs_base.json")

    logs_path = "/content/drive/My Drive/Colab Notebooks/logs/ljs_base/"
    pth_files = [f for f in os.listdir(logs_path) if f.startswith("G_")]
    latest_pth = max(pth_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    path = os.path.join(logs_path, "G_{}.pth".format(global_step)))

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(path, net_g, None)

    stn_tst = get_text("Hello and, again, welcome to the Aperture Science computer-aided enrichment center.", hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
            0, 0].data.cpu().float().numpy()

    # Save generated audio
    audio_save_path = "/content/drive/My Drive/Colab Notebooks/logs/ljs_base/test_wave/"
    audio_name = "G_{}.wav".format(global_step))
    audio_path = os.path.join(audio_save_path, audio_name)
    write(audio_path, hps.data.sampling_rate,
          audio)
    #Audio(audio, rate=hps.data.sampling_rate, normalize=False)
    print(audio_name, '오디오 생성을 완료하였습니다.')
    
