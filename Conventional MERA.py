import numpy as np
import torch
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
from lib.ncon import ncon
from lib import functional as func
import math
import time

def TensorExpand(A, chivec):
  """ expand tensor dimension by padding with zeros """

  if [*A.shape] == chivec:
    return A
  else:
    for k in range(len(chivec)):
      if A.shape[k] != chivec[k]:
        indloc = list(range(-1, -len(chivec) - 1, -1))
        indloc[k] = 1
        A = ncon([A, np.eye(A.shape[k], chivec[k])], [indloc, [1, -k - 1]])

    return A

def ascending(ham, w, u):
    legs1 = [[5, 9, 3, -3], [4, 11, 6, -4], [5, 7, 1, -1], [2, 11, 6, -2], [8, 12, 1, 2], [7, 8, 9, 10], [10, 12, 3, 4]]
    legs2 = [[5, 7, 3, -3], [4, 8, 6, -4], [5, 7, 1, -1], [2, 8, 6, -2], [9, 10, 1, 2], [9, 10, 11, 12], [11, 12, 3, 4]]
    legs3 = [[5, 7, 3, -3], [4, 12, 6, -4], [5, 7, 1, -1], [2, 10, 6, -2], [8, 9, 1, 2], [9, 10, 11, 12], [8, 11, 3, 4]]
    order1 = [9, 12, 5, 7, 11, 6, 3, 8, 10, 1, 2, 4]
    order2 = [11, 12, 9, 10, 8, 6, 5, 7, 3, 1, 2, 4]
    order3 = [5, 7, 4, 12, 11, 2, 6, 8, 9, 10, 3, 1]
    tensors = [w, w, w.conj(), w.conj(), u.conj(), ham, u]
    
    ham_out = ncon(tensors, legs1, order1) + ncon(tensors, legs2, order2) + ncon(tensors, legs3, order3)

    return ham_out

def descending(ro, w, u):
    legs1 = [[3, 4, 1, 2], [9, -3, 7, 1], [8, 11, 10, 2], [9, -1, 5, 3], [6, 11, 10, 4], [-4, 12, 7, 8], [-2, 12, 5, 6]]
    legs2 = [[3, 4, 1, 2], [9, 11, 7, 1], [8, 12, 10, 2], [9, 11, 5, 3], [6, 12, 10, 4], [-3, -4, 7, 8], [-1, -2, 5, 6]]
    legs3 = [[3, 4, 1, 2], [9, 11, 7, 1], [8, -4, 10, 2], [9, 11, 5, 3], [6, -2, 10, 4], [12, -3, 7, 8], [12, -1, 5, 6]]
    order1 = [11, 10, 4, 2, 8, 1, 7, 6, 12, 3, 9, 5]
    order2 = [9, 11, 3, 1, 12, 10, 4, 2, 5, 6, 7, 8]
    order3 = [9, 11, 3, 1, 4, 5, 6, 7, 12, 2, 10, 8]
    tensors = [ro, w, w, w.conj(), w.conj(), u, u.conj()]
    
    ro_out = (ncon(tensors, legs1, order1) + ncon(tensors, legs2, order2) + ncon(tensors, legs3, order3)) / 3
    
    return ro_out

def topdense(ro, w, u, sciter):
    chi = w.shape[3]
    if sciter == 0:
        legs1 = [[5, -7, 3, -3], [4, 7, 6, -4], [5, -5, 1, -1], [2, 7, 6, -2], [-8, 8, 3, 4], [-6, 8, 1, 2]]
        legs2 = [[5, 7, 3, -3], [4, 8, 6, -4], [5, 7, 1, -1], [2, 8, 6, -2], [-7, -8, 3, 4], [-5, -6, 1, 2]]
        legs3 = [[5, 7, 3, -3], [4, -8, 6, -4], [5, 7, 1, -1], [2, -6, 6, -2], [8, -7, 3, 4], [8, -5, 1, 2]]
        order1 = [7, 6, 5, 8, 4, 2, 3, 1]
        order2 = [5, 7, 1, 8, 6, 4, 3, 2]
        order3 = [8, 6, 5, 7, 3, 1, 4, 2]
        tensors = [w, w, w.conj(), w.conj(), u, u.conj()]
        
        op = (ncon(tensors, legs1, order1) + ncon(tensors, legs2, order2) + ncon(tensors, legs3, order3)) / 3
        #op = op - ncon([ro, np.eye(chi ** 2).reshape(chi, chi, chi, chi)], [[-1, -2, -3, -4],[-5, -6, -7, -8]])
        op = op.reshape(chi ** 4, chi ** 4)
        vtmp, ro_tmp = eigs(op, k=chi ** 4 - 1, which='LM')
        #print(vtmp)
        ro_tmp = np.abs(ro_tmp[0])
        ro_tmp = ro_tmp.reshape(chi, chi, chi, chi)
        ro = 0.5 * (ro_tmp + np.conj(ro_tmp.transpose(2, 3, 0, 1))) / ncon([ro_tmp], [[1, 2, 1, 2]])

    else:
        for _ in range(sciter):
            ro_tmp = descending(ro, w, u)
            ro = 0.5 * (ro_tmp + np.conj(ro_tmp.transpose(2, 3, 0, 1))) / ncon([ro_tmp], [[1, 2, 1, 2]])
            
    return ro


def env_w(ham, ro, w, u):
    legs11 = [[2, 3, -4, 1], [6, 11, 7, 1], [-1, 8, 4, 2], [5, 11, 7, 3], [10, 12, -3, 6], [9, 12, 4, 5], [8, 9, -2, 10]]
    legs12 = [[2, 3, -4, 1], [6, 8, 7, 1], [-1, -2, 4, 2], [5, 8, 7, 3], [11, 12, -3, 6], [9, 10, 4, 5], [9, 10, 11, 12]]
    legs13 = [[2, 3, -4, 1], [6, 12, 7, 1], [-1, -2, 4, 2], [5, 10, 7, 3], [8, 11, -3, 6], [8, 9, 4, 5], [9, 10, 11, 12]]
    legs21 = [[2, 3, 1, -4], [7, 10, 6, 1], [7, 8, 4, 2], [5, -2, -3, 3], [11, 12, 6, -1], [9, 12, 4, 5], [8, 9, 10, 11]]
    legs22 = [[2, 3, 1, -4], [7, 8, 6, 1], [7, 8, 4, 2], [5, -2, -3, 3], [11, 12, 6, -1], [9, 10, 4, 5], [9, 10, 11, 12]]
    legs23 = [[2, 3, 1, -4], [7, 8, 6, 1], [7, 8, 4, 2], [5, 11, -3, 3], [9, 12, 6, -1], [9, 10, 4, 5], [10, 11, 12, -2]]
    order11 = [9, 11, 7, 10, 12, 3, 1, 6, 5, 2, 4, 8]
    order12 = [9, 10, 11, 12, 8, 7, 3, 1, 6, 5, 2, 4]
    order13 = [6, 5, 10, 9, 12, 7, 8, 11, 3, 1, 2, 4]
    order21 = [11, 12, 9, 8, 4, 7, 10, 6, 2, 1, 3, 5]
    order22 = [7, 8, 9, 10, 11, 12, 2, 1, 6, 4, 3, 5]
    order23 = [7, 8, 2, 1, 6, 4, 9, 12, 10, 3, 5, 11]
    tensors = [ro, w, w.conj(), w.conj(), u, u.conj(), ham.conj()]
    
    env = ncon(tensors, legs11, order11) + ncon(tensors, legs12, order12) + ncon(tensors, legs13, order13) + ncon(tensors, legs21, order21) + ncon(tensors, legs22, order22) + ncon(tensors, legs23, order23)

    return env

def env_u(ham, ro, w, u):
    legs1 = [[3, 4, 1, 2], [7, 11, -3, 1], [-4, 12, 8, 2], [7, 9, 5, 3], [6, 12, 8, 4], [10, -2, 5, 6], [9, 10, 11, -1]]
    legs2 = [[3, 4, 1, 2], [7, 9, -3, 1], [-4, 10, 8, 2], [7, 9, 5, 3], [6, 10, 8, 4], [11, 12, 5, 6], [11, 12, -1, -2]]
    legs3 = [[3, 4, 1, 2], [7, 9, -3, 1], [-4, 12, 8, 2], [7, 9, 5, 3], [6, 11, 8, 4], [-1, 10, 5, 6], [10, 11, -2, 12]]
    order1 = [11, 12, 8, 4, 2, 7, 9, 10, 5, 3, 1, 6]
    order2 = [10, 8, 4, 2, 7, 9, 3, 1, 6, 5, 11, 12]
    order3 = [7, 9, 3, 1, 6, 11, 10, 4, 5, 2, 8, 12]
    tensors = [ro, w, w, w.conj(), w.conj(), u.conj(), ham.conj()]
    
    env = ncon(tensors, legs1, order1) + ncon(tensors, legs2, order2) + ncon(tensors, legs3, order3)

    return env

def updateSVD(wIn, leftnum):
    wSh = wIn.shape
    ut, st, vht = LA.svd(wIn.reshape(np.prod(wSh[0:leftnum:1]), np.prod(wSh[leftnum:len(wSh):1])), full_matrices=False)
    return - (ut @ vht).reshape(wSh)

def Ising(g=1.0):
    sX = np.array([[0, 1.0], [1.0, 0]])
    sZ = np.array([[1.0, 0], [0, -1.0]])
    H = np.kron(sX, sX) + 0.5 * g * (np.kron(sZ, np.eye(2)) + np.kron(np.eye(2), sZ))
    return H.reshape(2, 2, 2, 2)

def HeisenbergXY(g=1.0):
    sX = np.array([[0, 1.0], [1.0, 0]])
    sY = np.array([[0, -1.0], [1.0, 0]])
    H = np.kron(sX, sX) - g * np.kron(sY, sY)
    return H.reshape(2, 2, 2, 2)

def HeisenbergXXZ(g=1.0):
    sX = np.array([[0, 1.0], [1.0, 0]])
    sY = np.array([[0, -1.0], [1.0, 0]])
    sZ = np.array([[1.0, 0], [0, -1.0]])
    H = np.kron(sX, sX) - np.kron(sY, sY) + g * np.kron(sZ, sZ)
    return H.reshape(2, 2, 2, 2)

def doVarMERA(ham, ro, w, u, chi_max, epoch_size=2000, totlv=3, sciter=4, time_start=0, error_list=[], time_list=[], count=0, model='Ising'):
    for k in range(totlv - len(w) - 1):
        w.append(w[-1])
        u.append(u[-1])
    for k in range(totlv - len(ham)):
        ham.append(ham[-1])
        ro.append(ro[-1])
    chi = np.zeros(totlv + 1, dtype=int)
    chi[0] = ham[0].shape[0]
    for k in range(totlv):
        chi[k + 1] = min(chi_max, chi[k]** 3)
        w[k] = TensorExpand(w[k], [chi[k], chi[k], chi[k], chi[k + 1]])
        u[k] = TensorExpand(u[k], [chi[k], chi[k], chi[k], chi[k]])
        ham[k + 1] = TensorExpand(ham[k + 1], [chi[k + 1], chi[k + 1], chi[k + 1], chi[k + 1]])
        ro[k + 1] = TensorExpand(ro[k + 1], [chi[k + 1], chi[k + 1], chi[k + 1], chi[k + 1]])
    hamstart = ham[0]
    bias = max(LA.eigvalsh(ham[0].reshape(chi[0]** 2, chi[0]** 2)))
    ham[0] = ham[0] - bias * np.eye(chi[0]** 2).reshape(chi[0], chi[0], chi[0], chi[0])
    
    for epoch in range(epoch_size):
        ro[totlv] = topdense(ro[totlv], w[totlv - 1], u[totlv - 1], sciter)

        for k in range(totlv - 1, -1, -1):
            ro[k] = descending(ro[k + 1], w[k], u[k])
            #print(ncon([ro[k]], [[1, 2, 1, 2]]))
            #print(ncon([w[k],w[k].conj()],[[1,2,3,4],[1,2,3,4]]), ncon([u[k],u[k].conj()],[[1,2,3,4],[1,2,3,4]]))

        for k in range(totlv):
            wEnv = env_w(ham[k], ro[k + 1], w[k], u[k])
            w[k] = updateSVD(wEnv, 3)
            uEnv = env_u(ham[k], ro[k + 1], w[k], u[k])
            u[k] = updateSVD(uEnv, 2)
            ham[k + 1] = ascending(ham[k], w[k], u[k])

        if epoch % 5 == 0:
            energy = ncon([ro[0], ham[0]], [[1, 2, 3, 4], [1, 2, 3, 4]]) + bias
            time_now = time.time()
            
            if model == 'XXZ':
                exact = 1 - 4 * np.log(2)
            elif model == 'Ising' or model == 'XY':
                exact = - 4/math.pi
            error_list.append(energy - exact)
            time_list.append(time_now - time_start)
            print('\repoch %d: time = %f, energy err = %e, chi = %f' % (count, time_list[-1], error_list[-1], chi_max), end='')

        count += 1

        # if epoch % 100 == 0:
        #     ScSuper1 = ncon([w[-1], np.conj(w[-1])], [[1, -4, 2, -2], [1, -3, 2, -1]]).reshape(chi[-1]** 2, chi[-2]** 2)
        #     dtemp, utemp = eigs(ScSuper1, k=6, which='LM')
        #     scDims = -np.log(abs(dtemp)) / np.log(3) 
        #     print(scDims)
    print('')
            
    ham[0] = hamstart

    return ham, ro, w, u, error_list, time_list, count

def conventional_Mera_ternary(chi_list, epoch_list, totlv, H0, sciter, model='Ising'):

    chi = np.zeros(totlv + 1, dtype=int)
    w = [0] * totlv
    u = [0] * totlv
    ham = [0] * (totlv + 1)
    ham[0] = H0
    chi[0] = ham[0].shape[0]
    ro = [0] * (totlv + 1)
    ro[0] = np.eye(chi[0]** 2).reshape(chi[0], chi[0], chi[0], chi[0])

    error_list = []
    time_list = []
    count = 0
    reset = 0
    for k in range(totlv):
        chi[k + 1] = min(chi_list[0], chi[k]** 3)
        w[k] = np.random.rand(chi[k]** 3, chi[k + 1])
        w[k], _, _ = LA.svd(w[k], full_matrices=False)
        w[k] = w[k].reshape(chi[k], chi[k], chi[k], chi[k + 1])
        u[k] = (np.eye(chi[k]** 2, chi[k]** 2)).reshape(chi[k], chi[k], chi[k], chi[k])
        ro[k + 1] = np.eye(chi[k + 1]** 2).reshape(chi[k + 1], chi[k + 1], chi[k + 1], chi[k + 1])
        ham[k + 1] = np.zeros((chi[k + 1], chi[k + 1], chi[k + 1], chi[k + 1]))

    time_start = time.time()

    for i in range(len(epoch_list)):
        ham, ro, w, u, error_list, time_list, count = doVarMERA(ham, ro, w, u, chi_list[i], totlv=totlv, sciter=sciter, epoch_size=epoch_list[i], time_start=time_start, error_list=error_list, time_list=time_list, count=count, model=model)
        Mera = {'H':ham, 'ro':ro, 'w':w, 'u':u}
        if i == 1 and error_list[-1] > 2e-3 and model == 'Ising':
            reset = 1
            break

    return Mera, error_list, time_list, reset


def single_repeat(model='Ising', repeat=4):
    chi_list=[4, 6, 7, 8, 9, 10]
    epoch_list=[200, 500, 1000, 2000, 3000, 4000]
    # chi_list=[4, 6]
    # epoch_list=[200, 500]
    totlv = 3
    sciter = 4
    g = 1.0
    if model == 'Ising':
        H0 = Ising(g)
    elif model == 'XY':
        H0 = HeisenbergXY(g)
    elif model == 'XXZ':
        H0 = HeisenbergXXZ(g)
    # repeat = 4
    Mera_list = []
    error_list = []
    time_list = []
    sc_list = []

    i = 0
    while i < repeat:
        Mera, errors, times, reset = conventional_Mera_ternary(chi_list, epoch_list, totlv, H0, sciter, model=model)
        if reset:
            print('----------------Restart-------------------')
            continue
        i += 1
        Mera_list.append(Mera)
        error_list.append(errors)
        time_list.append(times)
        sc = func.mera_sc(torch.Tensor(Mera['w'][-1]))
        print('\n', sc)
        sc_list.append(sc)

    file_name = '.\\data\\MERA single\\' + model + ' Conventional ' + str(repeat) + '.npz'

    np.savez(file_name, 
        setting=np.array([totlv, repeat, g]),
        epoch=np.array(epoch_list),
        chi_list=np.array(chi_list),
        error_list=np.array(error_list),
        time_list=np.array(time_list),
        sc_list=np.array(sc_list))

if __name__ == "__main__":
    single_repeat('XY', repeat=3)
    single_repeat('XXZ', repeat=3)
    