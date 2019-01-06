from gym.envs.registration import register

register(
    id='timeseries-v0',
    entry_point='timeseries_ml_utils.envs:TsGym',
)
