import pytest

def test_construct():
    import bellini
    from bellini import Quantity, Species, Substance, Story
    import pint
    ureg = pint.UnitRegistry()

    s = Story()
    water = Species(name='water')
    s.one_water_quantity = bellini.distributions.Normal(
        loc=Quantity(3.0, ureg.mole),
        scale=Quantity(0.01, ureg.mole),
    )
    s.another_water_quantity = bellini.distributions.Normal(
        loc=Quantity(3.0, ureg.mole),
        scale=Quantity(0.01, ureg.mole),
    )
    s.combined_water = s.one_water_quantity + s.another_water_quantity
    s.combined_water.observed = True

    from bellini.api._numpyro import graph_to_numpyro_model
    model = graph_to_numpyro_model(s.g)

    from numpyro.infer.util import initialize_model
    import jax
    model_info = initialize_model(
        jax.random.PRNGKey(2666),
        model,
    )
    from numpyro.infer.hmc import hmc
    from numpyro.util import fori_collect

    init_kernel, sample_kernel = hmc(model_info.potential_fn, algo='NUTS')
    hmc_state = init_kernel(model_info.param_info,
                         trajectory_length=10,
                         num_warmup=300)
    samples = fori_collect(0, 500, sample_kernel, hmc_state,
        transform=lambda state: model_info.postprocess_fn(state.z))
