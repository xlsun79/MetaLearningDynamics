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
Before running neural analysis, please download the neural spiking data ("wilbur20210408_spike_times_activeunits.pkl" inside the zip file) into the same folder from:
https://www.dropbox.com/scl/fi/ds2mfnc4r70njihqx4w34/NeuralDataSharing.zip?rlkey=gvb0dsvi9s2586a68ex3yu51x&st=rwxspizp&dl=0
Other files in the folder are used for running LFADS and please check out code and instructions at https://github.com/claytonwashington/foraging-neural-dynamics

### Operating System
Tested on **Ubuntu 20.04**. Expected to work on any Linux distribution, macOS, or Windows with Python 3.10+.
