import numpy as np
import torch

CLASS_RELATIVE_WEIGHTS = np.array([[1., 29.34146341, 601.5, ],
                                       [1., 10.46296296, 141.25, ],
                                       [1., 3.6539924, 43.68181818, ],
                                       [1., 1.89223058, 8.20652174, ],
                                       [1., 2.31736527, 5.60869565, ],
                                       [1., 19.46666667, 64.88888889, ],
                                       [1., 6.30674847, 18.69090909, ],
                                       [1., 2.92041522, 7.46902655, ],
                                       [1., 1.5144357, 2.00347222, ],
                                       [1., 3.43076923, 9.4893617, ],
                                       [1., 27.11363636, 132.55555556, ],
                                       [1., 10.5, 283.5, ],
                                       [1., 3.65267176, 35.44444444, ],
                                       [1., 2.05277045, 8.74157303, ],
                                       [1., 2.75333333, 6.88333333, ],
                                       [1., 14.59493671, 82.35714286, ],
                                       [1., 6.32926829, 23.59090909, ],
                                       [1., 2.82828283, 7.70642202, ],
                                       [1., 1.43367347, 1.92465753, ],
                                       [1., 3.57429719, 8.31775701, ],
                                       [1., 29.04878049, 85.07142857, ],
                                       [1., 11.31632653, 28.43589744, ],
                                       [1., 7.16083916, 12.96202532, ],
                                       [1., 6.25675676, 5.38372093, ],
                                       [1., 44.66666667, 92.76923077],
                                       ])

# CLASS_RELATIVE_WEIGHTS = torch.tensor([e / torch.sum(e) for e in CLASS_RELATIVE_WEIGHTS])

CLASS_LOGN_RELATIVE_WEIGHTS = 1 + np.e * np.log(CLASS_RELATIVE_WEIGHTS)
CLASS_LOGN_RELATIVE_WEIGHTS = torch.tensor([e / np.sum(e) * len(e) for e in CLASS_LOGN_RELATIVE_WEIGHTS])


# !TODO: Refactor
CLASS_RELATIVE_WEIGHTS_MIRROR = CLASS_RELATIVE_WEIGHTS.copy()
CLASS_RELATIVE_WEIGHTS_MIRROR[0:10] += CLASS_RELATIVE_WEIGHTS_MIRROR[10:20]
CLASS_RELATIVE_WEIGHTS_MIRROR[0:10] /= 2
CLASS_RELATIVE_WEIGHTS_MIRROR[10:20] = CLASS_RELATIVE_WEIGHTS_MIRROR[0:10]

CLASS_LOGN_RELATIVE_WEIGHTS_MIRROR = 1 + np.log(CLASS_RELATIVE_WEIGHTS_MIRROR)
CLASS_LOGN_RELATIVE_WEIGHTS_MIRROR_NORMALIZED = torch.tensor([e / np.sum(e) * len(e) for e in CLASS_LOGN_RELATIVE_WEIGHTS_MIRROR])
CLASS_LOGN_RELATIVE_WEIGHTS_MIRROR = torch.tensor(CLASS_LOGN_RELATIVE_WEIGHTS_MIRROR)

CLASS_NEG_VS_POS = torch.Tensor(
    [3.57439734e-02, 2.93902439e+01, 6.22000000e+02,
     1.02654867e-01, 1.05370370e+01, 1.54750000e+02,
     0.29656608, 3.73764259, 55.63636364,
     0.65033113, 2.12280702, 12.54347826,
     0.60981912, 2.73053892, 8.02898551,
     6.67808219e-02, 1.97666667e+01, 6.82222222e+01,
     0.21206226, 6.64417178, 21.65454545,
     0.47630332, 3.31141869, 10.02654867,
     1.15944541, 2.27034121, 3.32638889,
     0.39686099, 3.79230769, 12.25531915,
     4.44258173e-02, 2.73181818e+01, 1.37444444e+02,
     9.87654321e-02, 1.05370370e+01, 3.10500000e+02,
     0.30198537, 3.75572519, 45.14814815,
     0.60154242, 2.28759894, 13.,
     0.50847458, 3.15333333, 9.38333333,
     8.06591500e-02, 1.47721519e+01, 8.80000000e+01,
     0.20038536, 6.59756098, 27.31818182,
     0.48333333, 3.1952862, 10.43119266,
     1.21708185, 2.17857143, 3.26712329,
     0.4, 4.00401606, 10.64485981,
     4.61796809e-02, 2.93902439e+01, 8.80000000e+01,
     0.12353472, 11.71428571, 30.94871795,
     0.21679688, 7.71328671, 14.7721519,
     0.34557235, 7.41891892, 6.24418605,
     3.31674959e-02, 4.51481481e+01, 9.48461538e+01]
)

COMP_WEIGHTS = torch.Tensor([[1, 2, 4] for i in range(25)])

CLASS_RELATIVE_WEIGHTS_NORMALIZED = torch.tensor(CLASS_RELATIVE_WEIGHTS_MIRROR / np.expand_dims(np.sum(CLASS_RELATIVE_WEIGHTS_MIRROR, axis=1), -1) * 3)
#Clamp min weight
CLASS_RELATIVE_WEIGHTS_NORMALIZED[CLASS_RELATIVE_WEIGHTS_NORMALIZED < 0.1] = 0.1

pass