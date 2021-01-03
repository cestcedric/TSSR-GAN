import os
import signal
import torch

class GracefulKiller:
    def __init__(self, save_path):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.save_path = save_path
        self.generator = None
        self.discriminator = None

    def attach_generator(self, generator):
        self.generator = generator

    def attach_discriminator(self, discriminator):
        self.discriminator = discriminator

    def exit_gracefully(self, signum, frame):
        if not self.generator == None:
            genpath = os.path.join(self.save_path, 'GEN_termination.pth')
            torch.save(self.generator.state_dict(), genpath)
            print('Generator state saved:', genpath)
        if not self.discriminator == None:
            dispath = os.path.join(self.save_path, 'DIS_termination.pth')
            torch.save(self.discriminator.state_dict(), dispath)
            print('Discriminator state saved:', dispath)
        print('Training killed gracefully :)')
        exit(0)