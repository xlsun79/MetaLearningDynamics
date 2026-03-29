# MetaLearningDynamics
Custom code for central analysis in Sun*, Comrie*, et al., 2026.

### Package installation
Before running the analysis, please cd into the MetaLearningDynamics folder, and install the environment by:
```
mamba env create -f environment.yml
```

Then activate the environment by:
```
mamba activate spyglass_metalearning
```

### Example data
Behavioral dataframes of an example dataset is saved in folder "example_data". 
Before running neural analysis, please download the neural spiking data ("wilbur20210408_spike_times_activeunits.pkl") into the same folder from:
https://www.dropbox.com/scl/fo/s01hq7l11j9z9o0odtozq/AAMV1LoJeogNxX-sifsqd-0?rlkey=l0visrjoqrsej5hljw6pwxa7v&st=30zezsq9&dl=0
Other files in the folder are used for running LFADS and please check out code and instructions at https://github.com/claytonwashington/foraging-neural-dynamics

### Operating System
Tested on **Ubuntu 20.04**. Expected to work on any Linux distribution, macOS, or Windows with Python 3.10+.
