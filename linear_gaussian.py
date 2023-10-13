import os
import json

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from utils.experiment import save_x_y_data
from utils.utils import (
    local_norm_sq,
    create_synthetic_data,
    empirical_risk,
    resample_classes,
    set_seed, 
    setup_checkpoint_dir, 
)
from utils.plot_utils import (
    get_boundary_function, 
    check_save_fig
)

from models.prediction_models import LinearDiscriminantAnalysis


checkpoint_dir = setup_checkpoint_dir("linear_gaussian")

mean_1 = np.array([1, 1])
mean_2 = np.array([-1, -1])
cov = np.array([[1, 0.5], [0.5, 1]])
cov_inv = np.linalg.inv(cov)

N = 1000
frac = 0.5
seed = 1502

set_seed(seed)

# Save all the parameters if the experiment 
with open(os.path.join(checkpoint_dir, "experiment_parameters.json"), "w") as f:
    parameters = {
        "mean_1": mean_1.tolist(), 
        "mean_2": mean_2.tolist(), 
        "cov": cov.tolist(), 
        "N": N, 
        "frac": frac,
        "seed": seed
    }
    json.dump(parameters, f, indent=4)



num = local_norm_sq(mean_1, mean_2, cov_inv)
bayes_risk = norm.cdf(-0.5 * np.sqrt(num))
print("Bayes risk", bayes_risk)

data = create_synthetic_data(mean_1, mean_2, cov, cov, N, frac=frac)
x = data[:, :-1]
y = data[:, -1]

lda = LinearDiscriminantAnalysis(mean_1, mean_2, cov, frac)

# Bayes optimal boundary is parametrized by beta
beta, c = lda.get_params()
print(beta, c)
y_hat = lda.predict(x)
print(sum(y_hat))
boundary = get_boundary_function(beta, c)

risk = empirical_risk(y, y_hat)
print(risk)


# After recourse is provided
x_after_recourse = lda.provide_recourse(x, y_hat)
y_hat_after_recourse = lda.predict(x_after_recourse)

y_after_recourse_x = y.copy()
y_after_recourse_x[y_hat == 0] = resample_classes(
    x_after_recourse[y_hat == 0], 
    lda.predict_proba
)
y_after_recourse_x_0 = y.copy()
y_after_recourse_x_0[y_hat ==0] = resample_classes(
    x[y_hat == 0], 
    lda.predict_proba
)

risk_after_recourse_x = empirical_risk(y_after_recourse_x, y_hat_after_recourse)
risk_after_recourse_x_0 = empirical_risk(y_after_recourse_x_0, y_hat_after_recourse)
print(risk_after_recourse_x)
# Plotting the result
fig, axs = plt.subplots(1, 2, figsize=(18, 7))
title_1 = f"Original distributions, Empirical Risk {risk:.4f}"
title_2 = f"Q(y|x, x_0) = P(y | x), Empirical Risk {risk_after_recourse_x:.4f}"
# title_3 = f"Q(y|x, x_0) = P(y |x_0), Empirical Risk {risk_after_recourse_x_0:.4f}"
lda.plot(x, y, y_hat, axs[0], title_1)
xyz_cond_probs = lda.plot(
    x_after_recourse, 
    y_after_recourse_x, 
    y_hat_after_recourse, 
    axs[1], 
    title_2
    )
db_coords = xyz_cond_probs[
    np.isclose(xyz_cond_probs[:, 2], .5), :][:, 0:2]
# lda.plot(x, y_after_recourse_x_0, y_hat_after_recourse, axs[1, 1], title_3)

fig.tight_layout()

check_save_fig(fig, checkpoint_dir, "linear_gaussian")

# Saving all data for possible post processing

data_dir = os.path.join(checkpoint_dir, 'data')
save_x_y_data(data_dir, x, y, y_hat)
save_x_y_data(
    data_dir, 
    x_after_recourse, 
    y_after_recourse_x, 
    y_hat_after_recourse,
    recourse=True)
with open(os.path.join(data_dir, "risk_before_after.dat",), 'w') as f:
    f.write(f"risk {risk} {risk_after_recourse_x}\n")
np.savetxt(os.path.join(data_dir, "xyz_cond_probs.dat"), xyz_cond_probs, fmt='%.6f')
np.savetxt(os.path.join(data_dir, "db_coords.dat"), db_coords, fmt='%.6f')
