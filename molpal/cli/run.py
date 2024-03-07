from argparse import ArgumentParser
import datetime
import os, logging
import signal
import sys, shutil
from timeit import default_timer as time
import pandas as pd

import ray

from molpal import Explorer
from molpal.cli.args import add_args, clean_and_fix_args

logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

def sigterm_handler(signum, frame):
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)

def topK_rate(k, top_explored, filename):
    top_explored = [x[0] for x in top_explored]
    labeled_fname = filename.split('.')[0].replace('libraries', 'data') + '_scores.csv.gz'
    df = pd.read_csv(labeled_fname, compression='gzip', header=0)

    if len(top_explored) < k:
        print('Top molecules explored is less than specified (%d), possibly due to limited iterations'%(k))
        print('Hit rate based on top-%d instead of top-%d'%(len(top_explored, k)))
        k = len(top_explored)
    real_topK = set(df.nsmallest(k, 'score')['smiles'])
    hit = 0
    for i in top_explored:
        if i in real_topK:
            hit +=1
    return hit, k

def main(args):
    print(
        """\
***********************************************************************
*   __    __     ______     __         ______   ______     __         *
*  /\ "-./  \   /\  __ \   /\ \       /\  == \ /\  __ \   /\ \        *
*  \ \ \-./\ \  \ \ \/\ \  \ \ \____  \ \  _-/ \ \  __ \  \ \ \____   *
*   \ \_\ \ \_\  \ \_____\  \ \_____\  \ \_\    \ \_\ \_\  \ \_____\  *
*    \/_/  \/_/   \/_____/   \/_____/   \/_/     \/_/\/_/   \/_____/  *
*                                                                     *
***********************************************************************"""
    )
    print("Welcome to MolPAL!")

    clean_and_fix_args(args)
    params = vars(args)

    print("MolPAL will be run with the following arguments:")
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")
    print(flush=True)

    # try:
    #     if "ip_head" in os.environ:
    #         ray.init(os.environ["ip_head"])
    #     else:
    #         ray.init("auto")
    # except ConnectionError:
    #     ray.init(num_cpus=params['ncpu'])

    # print("Ray cluster online with resources:")
    # print(ray.cluster_resources())
    # print(flush=True)

    path = params.pop("output_dir")
    explorer = Explorer(path, **params)
    if params['clear_tensorboard_dir']:
        if os.path.exists(f"{path}/log"):
            shutil.rmtree(f"{path}/log", ignore_errors=True)

    start = time()
    try:
        explorer.run()
    except BaseException:
        d_chkpts = f"{path}/chkpts"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        state_file = explorer.checkpoint(f"{d_chkpts}/iter_{explorer.iter}_{timestamp}")
        print(f'Exception raised! Intemediate state saved to "{state_file}"')
        raise
    stop = time()
    print('##########################')
    m, s = divmod(stop - start, 60)
    h, m = divmod(int(m), 60)
    d, h = divmod(h, 24)
    print(f"Total time for exploration: {d}d {h}h {m}m {s:0.2f}s")
    # print('##########################')
    # if params['objective'] == 'lookup':
    #     if params['k'] > 1:
    #         k = params['k']
    #     else:
    #         k = int(k * explorer.full_pool_size)
                                
    #     hit, k = topK_rate(k, explorer.top_explored(k), params['libraries'][0])
    #     print('Top-%d molecules retrieval rate: %.3f%%'%(k, hit/k*100))
    # print('##########################')
    print("Thanks for using MolPAL!")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args)
