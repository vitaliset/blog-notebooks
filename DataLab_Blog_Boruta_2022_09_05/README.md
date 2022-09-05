# Blog DataLab: Boruta

In this post, we discuss the need for variable selection, build the intuition behind boruta, and give practical usage tips.

## Installing the enviroment for the blog notebook (AKA having reproducible results)

Once you are inside this folder, you should create a virtual environment and install the same libraries used at this notebook. We can create de virtual enviroment using:
```console
conda create --name blog_datalab_boruta python=3.9.6 poetry=1.1.7 notebook=6.4.8
```

Then, we should enter the enviroment with:
```console
conda activate blog_datalab_boruta
```

Make sure you are inside the `20220620_blog_datalab_boruta` folder and can see the `pyproject.toml` (here you have the list of libraries we'll be using in this enviroment). You can install the libraries I'm using with poetry as follows:
```console
poetry install
```

Then we need to make sure the jupyter notebook will be have the option of looking at our virtual enviroment kernel:
```console
python -m ipykernel install --user --name=blog_datalab_boruta
```

We can then launch the jupyter notebook:
```console
jupyter notebook
```
And open the notebook making sure we are using the `blog_datalab_boruta` kernel.

## Once you are done with this post

After getting out of the jupyter notebook with `Ctrl + C`, if you want erase the virtual enviroment from your system you should run this command so the jupyter notebook won't be looking at the enviroment anymore:
```console
jupyter-kernelspec uninstall blog_datalab_boruta
```

We can then safelly get out of the virtual enviroment:
```console
conda deactivate
```

And finally remove it from the system:
```console
conda env remove --name blog_datalab_boruta
```








