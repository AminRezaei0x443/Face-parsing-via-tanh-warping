from torchvision.transforms import transforms
from model_lightning import Hybird
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from collections import OrderedDict
from preprocess import Warping, PrepareLabels,ToTensor
import torch


hparam = OrderedDict()
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=list, default=[5],
                    help='Select gpus')
parser.add_argument('--save-path', metavar='DIR', default="checkpoints", type=str,
                    help='path to save output')
parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                    help='supports three options dp, ddp, ddp2')
parser.add_argument("--optim", default=0, type=int, help="Optimizer: 0: Adam, 1: SGD, 2:SGD with Nesterov")
parser.add_argument('--precision', default=32, type=int,
                    help='Use 32bit or 16 bit precision')
parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help='Cudann benchmark')
parser.add_argument('--seed', type=int, default=42,
                    help='seed for initializing training. ')
parser.add_argument('--amp_level', type=str, default='O1', choices=('O0', 'O1', 'O2', 'O3'),
                    help='amp_level')
parser.add_argument("--accumulate", default=2, type=int, help="Accumulate_grad_batches")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size to use during training.")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--decay", default=0.01, type=float, help="Weight decay")
parser.add_argument("--dampening", default=0, type=float, help="dampening for momentum")
parser.add_argument("--workers", default=0, type=int, help="DataLoader Threads")

root_dir = {
    'train': "/content/warped_data/train",
    'val': "/content/warped_data/val",
    'test': "/content/warped_data/test"
}

preprocess_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_list = {
    'train':
        transforms.Compose([
            PrepareLabels((128, 128)),
            ToTensor()
        ]),
    'val':
        transforms.Compose([
            PrepareLabels((128, 128)),
            ToTensor()
        ]),
    'test':
        transforms.Compose([
            PrepareLabels((128, 128)),
            ToTensor()
        ])
}

pretrain_path = ""

hparam['args'] = parser.parse_args()
hparam['root_dir'] = root_dir
hparam['transforms'] = transforms_list
hparam['pretrain_path'] = pretrain_path
# print(hparam)

stage2 = Hybird(hparam)

checkpoint_callback = ModelCheckpoint(
    filepath='stage2/weights.ckpt',
    verbose=True,
    monitor='val_accu',
    mode='max'
)


trainer = pl.Trainer(checkpoint_callback=checkpoint_callback,
    gpus=torch.cuda.device_count(),
    max_epochs=hparam['args'].epochs,
    precision=hparam['args'].precision,
    benchmark=hparam['args'].benchmark,
    accumulate_grad_batches=hparam['args'].accumulate,
    amp_level=hparam['args'].amp_level))
trainer.fit(stage2)
