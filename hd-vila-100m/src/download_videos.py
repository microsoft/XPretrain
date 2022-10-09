from joblib import Parallel, delayed
import multiprocessing
import youtube_dl
import jsonlines
import json
import argparse
from tqdm import tqdm
import time
import os
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    parser.add_argument('--workdir', default='./hdvila_100m',type=str, help='Working Directory')
    parser.add_argument('--metafile', default='./meta_part0.jsonl', type=str, help='youtube video meta')
    parser.add_argument('--log', default='log_part0.log', type=str, help='log')
    parser.add_argument('--audio_only', action='store_true')
    args = parser.parse_args()
    return args


def check_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)


class YouTubeVideoDownloader():
    def __init__(self, metafile, workdir):
        self.videourls = self.readvideourls(metafile)
        self.workdir = workdir

    def readvideourls(self, metafile):
        vs = []
        with open(metafile,'r') as f:
            for l in jsonlines.Reader(f):
                vs.append(l['url'])
        logger.info('Number of videos to download: %d', len(vs))
        return vs

    def downloadvideo(self,vurl):
        format_id='22' # for 720p videos with audio
        if args.audio_only:
            format_id='140' # audio_only
        ydl_opts = {
            'outtmpl':os.path.join(self.workdir, 'download_videos') +'/%(id)s.%(ext)s',
            'merge_output_format':'mp4',
            'format':format_id, # 720P
            'skip_download':False,
            'ignoreerrors':True,
            'quiet':True
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            start = time.time()
            result = ydl.download([vurl])
            end = time.time()
            if result != 0:
                logger.error('Fail to download %s', vurl)
            logger.info('Time for download video %.2f sec', end-start)
            return result
            
    def downloadallParallel(self):
        num_cores = multiprocessing.cpu_count()
        logger.info(f"num cores: {num_cores}")
        results = Parallel(n_jobs=50, backend='threading')(delayed(self.downloadvideo)(v) for v in tqdm(self.videourls))
        results = [x for x in results if x is not None]
        logger.info(f"Number of videos downloaded: {len(results)}")


if __name__ == '__main__':
    args = parse_args()
    
    metafile = os.path.join(args.workdir, 'metafiles', args.metafile)
    logdir = os.path.join(args.workdir,'download_video_log')

    check_dirs(os.path.join(args.workdir, 'download_videos'))
    check_dirs(logdir)

    logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(logdir, args.log),
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.info(args)

    yvd = YouTubeVideoDownloader(metafile, args.workdir)
    yvd.downloadallParallel()