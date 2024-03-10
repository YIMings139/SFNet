from __future__ import absolute_import, division, print_function

from options import SFNetOptions
from trainer import Trainer

options = SFNetOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
