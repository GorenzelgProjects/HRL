# How to run training and testing

Everything is controlled by hydra. All the base configs can be seen [here](../config/config.yaml).
In order to override settings you can either run with an experiment file like [astar_example](../config/experiment/astar_example.yaml) by:
```bash
python main.py experiment=astar_example
```
and/or you can override using through arguments in CLI:
```bash
python main.py experiment=astar_example experiment.levels=[1,2,3]
```

_Note that any nested configs can be overridden using dots: `.`_
