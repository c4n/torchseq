import pdb
from sts_eval import STSEval
from proc_instance import InstanceProcessing

def main(args):
    # processing and get embedding
    get_vectors = InstanceProcessing(
        args.path_vocab,
        args.path_pretrained, 
        args.path_checkpoint, 
        args.device
    )
    sent_eval = STSEval(get_vectors, args.path_sts_data)
    results = sent_eval(args.transfer_tasks)
    return results

if __name__ == "__main__":
    class Args:
        # sta eval
        transfer_tasks = ['STS12']
        # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SST2', 'SST5', 'STSBenchmark']
        path_sts_data = '/home/weerayut/Workspace/projects/SentEval/data'
        
        # embedding
        device = 'cpu'
        path_vocab = "../data/"
        path_pretrained = "../data/"
        path_checkpoint = '../runs/vae/20220920_232720_paraphrasing_vae_mscoco_789'
    
    results = main(Args())
    # pdb.set_trace()