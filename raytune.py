from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
import main_noiser

def objective(metrics_dict)->float:
    # compute a score for the hparam set given the metrics dict
    pass
def trainable(config):
    # train noiser model with the given hparam config
    metrics_dict = main_noiser.main(config)
    # Compute hparam score
    score = objective(metrics_dict)
    # return score to ray tune manager
    return {"score": score}


"""
Examles of sampling spaces we can specify on the search space
config = {
    "uniform": tune.uniform(-5, -1),  # Uniform float between -5 and -1
    "quniform": tune.quniform(3.2, 5.4, 0.2),  # Round to multiples of 0.2
    "loguniform": tune.loguniform(1e-4, 1e-1),  # Uniform float in log space
    "qloguniform": tune.qloguniform(1e-4, 1e-1, 5e-5),  # Round to multiples of 0.00005
    "randn": tune.randn(10, 2),  # Normal distribution with mean 10 and sd 2
    "qrandn": tune.qrandn(10, 2, 0.2),  # Round to multiples of 0.2
    "randint": tune.randint(-9, 15),  # Random integer between -9 and 15
    "qrandint": tune.qrandint(-21, 12, 3),  # Round to multiples of 3 (includes 12)
    "lograndint": tune.lograndint(1, 10),  # Random integer in log space
    "qlograndint": tune.qlograndint(1, 10, 2),  # Round to multiples of 2
    "choice": tune.choice(["a", "b", "c"]),  # Choose one of these options uniformly
    "func": tune.sample_from(
        lambda spec: spec.config.uniform * 0.01
    ),  # Depends on other value
    "grid": tune.grid_search([32, 64, 128]),  # Search over all these values
}
"""

def tuner_main():
    """ See docs on https://docs.ray.io/en/latest/tune/key-concepts.html """
    # Define the search space
    search_space = {"reg_gen": tune.uniform(0, 1), 
                    "reg_uni": tune.uniform(0, 20)}

    algo = BayesOptSearch(random_search_steps=4)

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            search_alg=algo,
        ),
        run_config=train.RunConfig(stop={"training_iteration": 20}),
        param_space=search_space,
    )
    tuner.fit()

if __name__=="__main__":
    tuner_main()