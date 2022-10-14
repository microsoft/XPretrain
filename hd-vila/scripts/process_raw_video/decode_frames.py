import argparse
import os
os.system('pip install Pillow')
os.system('pip install decord')
import jsonlines
from tqdm import tqdm
import time
from PIL import Image
import decord
import multiprocessing
from joblib import Parallel, delayed
from glob import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='decode frames')
    parser.add_argument('--workdir', default='/data',type=str, help='workdir')
    parser.add_argument('--inputfile', default='train.jsonl', type=str, help='inputfile')
    parser.add_argument("--outputfile",type=str, default="train_result.jsonl", help="outputfile")

    args = parser.parse_args()
    return args

def check_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def load_clip_text(args):
    p = os.path.join(args.workdir,'lsmdc/', args.inputfile)
    data = []
    with open(p,'r') as f:
        for l in jsonlines.Reader(f):
            data.append(l)
    return data

def extract_single_clip(clip_text):
    try:
        clip_id = clip_text['clip_id']
        clip_path = os.path.join(args.workdir, 'lsmdc/videos/{}.avi'.format(clip_id))
        if os.path.exists(clip_path):
        
            out_folder = os.path.join(os.path.join(args.workdir, 'lsmdc/video_frames',clip_id))
            out_folder_lr = os.path.join(os.path.join(args.workdir, 'lsmdc/video_frames_lr',clip_id))
            os.system('rm -rf {}'.format(out_folder))
            
            check_dirs(out_folder)

            vr = decord.VideoReader(clip_path, ctx=decord.cpu(0))
            fps = vr.get_avg_fps()
            sample_id = np.round(np.linspace(0, len(vr)-1, round(len(vr)/fps*6))).astype(int)
            if len(sample_id)<=20:
                sample_id = np.round(np.linspace(0, len(vr)-1, 20)).astype(int)


            for i in range(len(sample_id)):
                frame = vr[sample_id[i]].asnumpy()
                img = Image.fromarray(frame).convert("RGB")
                img.save(os.path.join(out_folder, clip_id.split('/')[-1]+'_{0:03d}.jpg'.format(i)))

                img = img.resize((288,180),Image.BICUBIC)
                img.save(os.path.join(out_folder_lr, clip_id.split('/')[-1]+'_{0:03d}.jpg'.format(i)))

            return {'clip_id':clip_id, 'num_frame':len(sample_id)}
        else:
            return None
    except:
        return None

def main(args):

    clip_texts = load_clip_text(args)


    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    results = Parallel(n_jobs=2)(delayed(extract_single_clip)(c) for c in tqdm(clip_texts))
    results = [x for x in results if x is not None]

    print(len(results))
    check_dirs(os.path.join(args.workdir,'lsmdc/decode_results'))
    save_path = os.path.join(args.workdir,'lsmdc/decode_results',args.outputfile)
    print(save_path)
    with jsonlines.open(save_path, 'w') as f:
        for i in tqdm(range(len(results))):
            f.write(results[i])
    print('write done')


if __name__ =='__main__':
    args = parse_args()
    
    print(args.workdir)
    main(args)
