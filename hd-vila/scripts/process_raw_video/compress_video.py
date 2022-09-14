import os
import argparse
import subprocess
import time
from multiprocessing import cpu_count
import subprocess
import multiprocessing
from joblib import Parallel, delayed
import jsonlines
from tqdm import tqdm
from multiprocessing import Pool
from glob import glob
from decord import VideoReader, cpu


def parse_args():
    parser = argparse.ArgumentParser(description='video processing')
    parser.add_argument('--workdir', default='/data',type=str, help='work dir')
    parser.add_argument('--inputdir', default='datasets/msrvtt/videos', type=str, help='inputdir')
    parser.add_argument('--outputdir', default='datasets/msrvtt/videos_6fps', type=str, help='outputdir')
    parser.add_argument('--vidfile', default='datasets/msrvtt/train.jsonl', type=str, help='video id')
    args = parser.parse_args()
    return args

def check_dirs(dirs):
    if not os.path.exists(dirs):
        print(dirs)
        os.makedirs(dirs, exist_ok=True)


class CompressVideo():
    def __init__(self, vidfile, workdir, inputdir, outputdir):
        self.workdir = workdir
        self.vidfile = vidfile
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.vids = self.loadvids()

    def loadvids(self):
        vids = []
        with open(os.path.join(self.workdir,self.vidfile), 'r') as f:
            for l in jsonlines.Reader(f):
                vids.append(l)
        return vids

    def run(self, cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        return out.decode('utf-8')

    def compress_single_clip(self,data):
        vid = data['clip_id']

        input_video_path = os.path.join(self.workdir, self.inputdir, '{}.mp4'.format(vid))

        vr = VideoReader(input_video_path, ctx=cpu(0))
        time = len(vr) * vr.get_avg_fps()

        output_video_path = os.path.join(self.workdir,self.outputdir, vid+'.mp4')
        check_dirs(os.path.join(self.workdir,self.outputdir))

        cmd = ['ffmpeg',
                '-y',  # (optional) overwrite output file if it exists
                '-i', input_video_path,
                '-max_muxing_queue_size', '9999',
                '-r', '6',  
                output_video_path]


        self.run(cmd)

        if os.path.isfile(output_video_path):
            return vid + '*' + str(len(vr))
        else:
            return None


    def compress_clips(self):

        results = []
        print('start process')
        for vid in tqdm(self.vids):
            result =  self.compress_single_clip(vid)
            results.append(result)
        print(len(results))


    def compress_clips_parallel(self):
        num_cores = multiprocessing.cpu_count()
        print(num_cores)
        print('start process')
        results = Parallel(n_jobs=20, backend = 'threading')(delayed(self.compress_single_clip)(v) for v in tqdm(self.vids))

        results = [x for x in results if x is not None]

        print(len(results))


if __name__ == "__main__":

    args = parse_args()
    print(args)

    cpv = CompressVideo(args.vidfile, args.blob_mount_dir, args.inputdir, args.outputdir)
    cpv.compress_clips_parallel()
