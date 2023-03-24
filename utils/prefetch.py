import torch

class prefetcher():
    def __init__(self, loader, rank, mode):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.rank = rank
        self.mode = mode
        self.preload()

    def preload(self):
        if self.mode == 'train':
            try:
                self.next_sg1, self.next_sg2 = next(self.loader)
            except StopIteration:
                self.next_sg1 = None
                self.next_sg2 = None
                return
            with torch.cuda.stream(self.stream):
                self.next_sg1 = self.next_sg1.cuda(self.rank, non_blocking=True)
                self.next_sg2 = self.next_sg2.cuda(self.rank, non_blocking=True)
                self.next_sg1 = self.next_sg1.float()
                self.next_sg2 = self.next_sg2.float()
        else:
            try:
                self.next_input, self.next_target = next(self.loader)
            except StopIteration:
                self.next_input = None
                self.next_target = None
                return
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.cuda(self.rank, non_blocking=True)
                self.next_target = self.next_target.cuda(self.rank, non_blocking=True)
                self.next_input = self.next_input.float()
                self.next_target = self.next_target.long()
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input if self.mode == 'tune' else self.next_sg1
        target = self.next_target if self.mode == 'tune' else self.next_sg2
        self.preload()
        return input, target