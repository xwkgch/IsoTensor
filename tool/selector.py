import random
import torch.optim.lr_scheduler as Sch
class RandomSelect(object):
    def __init__(self, opt_list, stride=5):
        super().__init__()
        self.opt_list = opt_list
        self.stride = stride
        self.current = random.choice(self.opt_list)
        self.sch = []
        self.count = 0

    def select(self):
        self.count += 1

        if len(self.sch) > 0:
            self.sch[self.opt_list.index(self.current)].step()

        if len(self.opt_list) > 1 and self.count % self.stride == 0:
            for opt in self.opt_list:
                opt.param_groups[0]['lr'] = self.current.param_groups[0]['lr']

            self.current = random.choice(self.opt_list[1::])
        else:
            for opt in self.opt_list:
                opt.param_groups[0]['lr'] = self.current.param_groups[0]['lr']
            self.current = self.opt_list[0]


        return self.current
    
    def add_scheduler(self, type='StepLR', **kwds):
        for opt in self.opt_list:
            self.sch.append(Sch.StepLR(opt, step_size=10, gamma=0.999))