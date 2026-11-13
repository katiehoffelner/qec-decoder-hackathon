#this is a BB code
import numpy as np

def build_18_4_4_qldpc():
  
  # bb codes have l and m parameters
  l, m = 3,3

  def shift():
    Ix = np.roll(np.eye(ell, dtype=np.uint8), px, axis=1)
    Iy = np.roll(np.eye(m, dtype=np.uint8), py, axis=1)
    return np.kron(Ix, Iy)

    # identity and x and y shifts bc bb code properties i think
    I = shift_matrix(0, 0)
    x = shift_matrix(1, 0)
    y = shift_matrix(0, 1)
    xy = shift_matrix(1, 1)

    # polynomials A = 1 + x + xy  mod 2
    A = (I + x + xy) % 2
    # B = 1 + y + xy (mod 2)
    B = (I + y + xy) % 2

    # CSS parity-check matrices the hz dot hx = 0
    Hx = np.hstack([A, B])
    Hz = np.hstack([B.T, A.T])

    return Hx, Hz

Hx, Hz = build_18_4_4_qldpc()
