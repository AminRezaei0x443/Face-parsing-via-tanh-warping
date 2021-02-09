from torchvision.transforms import transforms
import torch
from model_lightning import Stage1
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from collections import OrderedDict
from preprocess import Warping

hparam = OrderedDict()
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', metavar='DIR', default="checkpoints", type=str,
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
parser.add_argument("--workers", default=4, type=int, help="DataLoader Threads")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs to train")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--decay", default=0.01, type=float, help="Weight decay")
parser.add_argument("--dampening", default=0, type=float, help="dampening for momentum")

root_dir = {
    'train': "/content/helenstar_release/train",
    'val': "/content/helenstar_release/train",
    'test': "/content/helenstar_release/test"
}
preprocess_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_list = {
    'train':
        transforms.Compose([
            Warping((512, 512), preprocess_device)
        ]),
    'val':
        transforms.Compose([
            Warping((512, 512), preprocess_device)
        ]),
    'test':
        transforms.Compose([
            Warping((512, 512), preprocess_device)
        ])
}

hparam['args'] = parser.parse_args()
hparam['root_dir'] = root_dir
hparam['transforms'] = transforms_list

print(hparam['args'])

checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/',
    verbose=True,
    monitor='val_loss',
    mode='min'
)

train_stage1 = Stage1(hparam)
trainer = pl.Trainer(
    checkpoint_callback=checkpoint_callback,
    gpus=torch.cuda.device_count(),
    max_epochs=hparam['args'].epochs,
    precision=hparam['args'].precision,
    benchmark=hparam['args'].benchmark,
    accumulate_grad_batches=hparam['args'].accumulate,
    amp_level=hparam['args'].amp_level
)
trainer.fit(train_stage1)
