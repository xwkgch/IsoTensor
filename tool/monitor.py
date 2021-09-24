import torch
import numpy as np
import time

class MERAMonitor(object):
    r"""Monitoring computation process of tensor network optimization. Timing starts at the instantiation.
    Args:

    """
    def __init__(self, Hamiltonian, Network):
        super().__init__()
        self.h = Hamiltonian
        self.net = Network
        self.E_error = 99999
        self.count = 0
        self.time_start=time.time()
        # print('\n')

    def display(self, loss, opt, stride=5, container=[]):
        r"""display the basic MERA optimization information
        """
        if self.count % stride == 0:
            if self.h.dtype == torch.cdouble:
                energy = loss.real.item() + np.real(self.h.bias)
            else:
                energy = loss.item() + self.h.bias
            
            self.E_error = abs(energy - self.h.E_exact)
            
            print('\repoch %d: energy err = %e, chi = %f, lr = %f, %s' % (self.count, self.E_error, self.net.chi[-1], opt.param_groups[0]['lr'], opt.label), end='')

            container.append(self.E_error)

        self.count += 1


    def end(self):
        r"""Timing over
        """
        time_end=time.time()
        print('\nTime used: ', time_end-self.time_start)

class TNRMonitor(object):
    r"""Monitoring computation process of tensor network optimization. Timing starts at the instantiation.
    Args:

    """
    def __init__(self, ten, Network):
        super().__init__()
        self.ten = ten
        self.net = Network
        self.count = 0
        self.time_start=time.time()
    
    def display(self, loss, lv, stride=5):
        r"""display the basic TNR optimization information
        """
        if self.count % stride == 0:
        
            print('\rLevel: %d -> epoch %d: proj err = %e' % (lv, self.count, loss.item()), end='')

        self.count += 1

    def nextlv(self):
        self.count = 0
        print('')

    def end(self):
        r"""Timing over
        """
        time_end=time.time()
        print('Time used: ', time_end-self.time_start)