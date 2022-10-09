import jsonlines
import os
from tqdm import tqdm
import logging
import argparse
import re
import subprocess
import multiprocessing
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(description='youtube video processing')
    parser.add_argument('--workdir', default='./hdvila_100m',type=str, help='Working Directory')
    parser.add_argument('--metafile', default='meta_part0.jsonl', type=str, help='youtube video meta')
    parser.add_argument('--resultfile', default='cut_part0.jsonl', type=str, help='processed videos')
    parser.add_argument('--log', default='log_part0.log', type=str, help='log')
    args = parser.parse_args()
    return args


def check_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)


class Cutvideos():
    def __init__(self, metafile, workdir, resultfile):
        self.workdir = workdir
        self.metafile = metafile
        self.resultfile = resultfile
        self.metas = self.loadmetas()

    def loadmetas(self):
        metas = []
        with open(self.metafile, 'r') as f:
            for l in jsonlines.Reader(f):
                metas.append(l)
        return metas

    def hhmmss(self, timestamp1, timestamp2):
        hh,mm,s = timestamp1.split(':')
        ss,ms = s.split('.')
        timems1 = 3600*1000*int((hh)) +  60*1000*int(mm) + 1000*int(ss) + int(ms)
        hh,mm,s = timestamp2.split(':')
        ss,ms = s.split('.')
        timems2 = 3600*1000*int((hh)) +  60*1000*int(mm) + 1000*int(ss) + int(ms)
        dur = (timems2 - timems1)/1000
        return str(dur)

    def run(self, cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        return out.decode('utf-8')

    def extract_single_clip(self,sb, in_filepath, out_filepath):
        cmd = ['ffmpeg', '-ss', sb[0], '-t', self.hhmmss(sb[0], sb[1]),'-accurate_seek', '-i', in_filepath, '-c', 'copy',
            '-avoid_negative_ts', '1', '-reset_timestamps', '1',
            '-y', '-hide_banner', '-loglevel', 'panic', '-map', '0',out_filepath]
        self.run(cmd)
        if not os.path.isfile(out_filepath):
            raise Exception(f"{out_filepath}: ffmpeg clip extraction failed")

    def extract_clips(self, meta):
        clips = meta['clip']
        vid = meta['video_id']
        outfolder = os.path.join(self.workdir,'video_clips', vid)
        check_dirs(outfolder)
        result = []
        # try:
        for c in clips:
            self.extract_single_clip(c['span'], os.path.join(self.workdir,'download_videos', vid + '.mp4'), os.path.join(outfolder, c['clip_id']))
            result.append(c['clip_id'])
        # except:
        #     pass

        return result

    def extract_all_clip(self):
        results = []
        for v in tqdm(self.metas):
            result = self.extract_clips(v)
            results.extend(result)

        logger.info(f"Number of clips processed: {len(results)}")
        with jsonlines.open(os.path.join(self.workdir, 'cut_video_results', self.resultfile), 'w') as f:
            for l in results:
                f.write(l)
        

if __name__ == '__main__':
    args = parse_args()
    
    metafile = os.path.join(args.workdir, 'metafiles', args.metafile)
    logdir = os.path.join(args.workdir,'cut_video_log')

    check_dirs(os.path.join(args.workdir, 'video_clips'))
    check_dirs(os.path.join(args.workdir, 'cut_video_results'))
    check_dirs(logdir)

    logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(logdir, args.log),
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.info(args)

    cvd = Cutvideos(metafile, args.workdir, args.resultfile)
    cvd.extract_all_clip()