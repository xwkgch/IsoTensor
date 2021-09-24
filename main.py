import torch
import example
import lib

lib.torchncon.ncon_check=False
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":

    # example.mera.construct_MERA()
    # example.mera.g_function()
    example.mera.compare_method()

    # example.meraad.construct_MERA_simple()

    # example.tnr.construct_TNR(chi=16,totlv=10,epoch=2000)
    # example.tnr.beta_function()
    # example.tnr.sc_calc()
    # example.tnr.compare_method()
