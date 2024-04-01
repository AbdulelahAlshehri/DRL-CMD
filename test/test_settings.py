
from config.run_settings import RunSettings, init_default_config
import yaml
'''
mbt:
  name: "MBT"
  building_blocks: [1,2,3,4,15,22,29,42,123]
  constraints:
    NUM_GROUPS:
      min: 3
      max: 8
    NUM_REPEAT_GROUPS:
      min: 0
      max: 7
    NUM_FUNC_GROUPS: 
      min: 1
      max: 6
    MOLECULAR_WEIGHT: 
      min: 80
      max: 200
    FLASH POINT:
      min: 273
      max: 393
    MELTING_POINT:
      min: 173
      max: 310
    BOILING_POINT:
      min: 373
      max: 600
    LC50:
      min: 0
      max: 4.8
    HSP:
      min: 18
      max: 21
  num_rings: all
  level: 1'''


def test_load_default_config():
    args = '-r 1 -a ppo a2c dqn'.split()
    settings = RunSettings(args)
    with open('config/config.yml', 'r') as f:
        expected = yaml.safe_load(f)

    assert settings.rings == 1
    assert settings.hp['num_iterations'] == expected['hp']['num_iterations']
    assert settings.hp['max_steps'] == expected['hp']['max_steps']
    assert settings.algorithms['enable_rf'] == False
    assert settings.algorithms['enable_dqn'] == True
    assert settings.logdir == './'

# def test_load_modified_config():
#     parser = create_parser()
#     args = parser.parse_args(['-c', 'mbt', 'absorbent',
#                               'surfactant', '-sp', '-a',  'ppo', 'rf',
#                               'a2c'])
#     settings = RunSettings(args)
