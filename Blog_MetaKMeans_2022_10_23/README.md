# Blog DataLab: MetaKMeans

This post discusses how one could implement an ensemble of K-Means aggregating K-Means with a meta cluster. We also discuss aspects of clustering as a whole briefly.

## Installing the enviroment for the blog notebook (AKA having reproducible results)

Once you are inside this folder, you should create a virtual environment and install the same libraries used at this notebook. We can create de virtual enviroment using:
```console
conda create --name blog_kmeans_kmeans python=3.9.6 poetry=1.1.7 notebook=6.4.8
```

Then, we should enter the enviroment with:
```console
conda activate blog_kmeans_kmeans
```

Make sure you are inside the `Blog_MetaKMeans_2022_10_23` folder and can see the `pyproject.toml` (here you have the list of libraries we'll be using in this enviroment). You can install the libraries I'm using with poetry as follows:
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
conda env remove --name blog_kmeans_kmeans
```








