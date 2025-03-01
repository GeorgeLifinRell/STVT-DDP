from STVT import datasets

def build_dataloader(args, distributed=True):
    return datasets.__dict__[args.dataset](args, distributed)
