import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

Temperature = 2
file_name = "BismuthDataT_{}K.txt".format(str(Temperature))
data = np.loadtxt(file_name)
t = data[:,[0]]
A_t = data[:,[1]]
A_t_err = data[:,[2]]

mask =  ( t < 10.0 )
n = np.size(t[mask])
X_train, y_train_noisy = t[mask], A_t[mask]
X_train = np.reshape(X_train, (n,1))

noise_std = 0.1 #0.1 0.2
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(
      kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
  )

gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X_train, return_std=True)

plt.errorbar(t[mask], A_t[mask], A_t_err[mask],fmt='.', label='data', color='black', elinewidth=2, capsize = 3)
plt.plot(t[mask], mean_prediction, label="GPR", color = 'red')
plt.fill_between(
    t[mask].ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:red",
    alpha=0.35,
    label=r"95% confidence interval",
)

plt.xlim(0,5)
plt.ylim(15,20)
plt.legend(loc=1,fontsize=10,handlelength=3)
plt.xlabel('t (micro-second)', fontsize = 15)
plt.ylabel('A (%)', fontsize = 15)
title_name = "T = {} Kelvin".format(str(Temperature))
plt.title(title_name)
output_name = "GPR_fit_err_{}.png".format(str(Temperature))
plt.savefig(output_name)
plt.clf()
