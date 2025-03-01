#from . import models
from STVT.models.STVT import STVT
from torch.nn.parallel import DistributedDataParallel as DDP

def build_model(args):
    try:
        print(f"Building model on rank {args.rank}")
        device = f"cuda:{args.rank}"
        model = STVT(dataset=args.dataset).to(device)
        model = DDP(model, device_ids=[args.rank])
        print(f"Model built on rank {args.rank}")
    except Exception as e:
        print(f"Model building failed on rank {args.rank}")
        print(e)
        return None
    return model
