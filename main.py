import torch
import example
import lib

lib.torchncon.ncon_check=False
torch.set_default_dtype(torch.float64)
 
if __name__ == "__main__":
    info = ' '
    lift = True
    if lift:
        chi_list=[4, 6, 7, 8, 9, 10]
        epoch_list=[200, 500, 1000, 2000, 3000, 4000]
    else:
        chi_list=[10, 10]
        epoch_list=[1000, 9000]
        info = info + 'nonlift'

    # info = info + 'RMSprop'
    # chi_list=[4, 6, 7]
    # epoch_list=[200, 500, 1000]

    example.mera.construct_MERA(chi_list=chi_list, epoch_list=epoch_list)
    # example.mera.compare_method()
    # example.mera.single_repeat(model='XY', chi_list=chi_list, epoch_list=epoch_list, info=info, mode='cuda', type='Mix', repeat=5)
    # example.mera.single_repeat(model='XXZ', chi_list=chi_list, epoch_list=epoch_list, inwfo=info, mode='cuda', type='Mix', repeat=5)
    # example.mera.single_repeat(model='XY', chi_list=chi_list, epoch_list=epoch_list, info=info, mode='cuda', type='EV', repeat=5)
    # example.mera.single_repeat(model='XXZ', chi_list=chi_list, epoch_list=epoch_list, info=info, mode='cuda', type='EV', repeat=5)
    # example.mera.single_repeat(model='XY', chi_list=chi_list, epoch_list=epoch_list, info=info, mode='cpu', type='EV', repeat=5)
    # example.mera.single_repeat(model='XXZ', chi_list=chi_list, epoch_list=epoch_list, info=info, mode='cpu', type='EV', repeat=5)
    # example.mera.single_repeat(model='XY', chi_list=chi_list, epoch_list=epoch_list, info=info, mode='cuda', type='Mix', repeat=8)
    # example.mera.single_repeat(model='XXZ', chi_list=chi_list, epoch_list=epoch_list, info=info, mode='cuda', type='Mix', repeat=8)
    # example.mera.single_repeat(model='XY', chi_list=chi_list, epoch_list=epoch_list, info=info, mode='cuda', type='EV', repeat=4)

    # example.tnr.construct_TNR(chi=8,totlv=8,epoch=4000)
    # example.tnr.beta_function()
    # example.tnr.sc_calc()
    # example.tnr.compare_method()

