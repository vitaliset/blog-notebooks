# Blog: Hyperparameters search with threshold-dependent metrics

This post discusses how dangerous it can be to do hyperparameter tunning with threshold-dependents metrics directly and how to workaround it.

## Installing the enviroment for the blog notebook (AKA having reproducible results)

Once you are inside this folder, you should create a virtual environment and install the same libraries used at this notebook. We can create de virtual enviroment using:
```console
conda create --name blog_threshold_dependent_opt python=3.9.6 poetry=1.3.1 notebook=6.4.8
```

If you get an `PackagesNotFoundError` for poetry, you may need to set a new channel for conda to install poetry (namely `conda-forge`) running:
```console
conda config --append channels conda-forge
```

Then, we should enter the enviroment with:
```console
conda activate blog_threshold_dependent_opt
```

Make sure you are inside the `Blog_Threshold_Dependent_Opt_2023_01_06` folder and can see the `pyproject.toml` (here you have the list of libraries we'll be using in this enviroment). You can install the libraries I'm using with poetry as follows:
```console
poetry install
```

We can then launch the jupyter notebook:
```console
jupyter notebook
```

## Once you are done with this post

After getting out of the jupyter notebook with `Ctrl + C`, if you want erase the virtual enviroment we can then safelly get out of the virtual enviroment:
```console
conda deactivate
```

And finally remove it from the system:
```console
conda env remove --name blog_threshold_dependent_opt
```








