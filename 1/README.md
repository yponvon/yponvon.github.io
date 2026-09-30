# CS180 Project 1: Images of the Russian Empire

Webpage: https://yponvon.github.io/1/

## Folders

```
1/
├── index.html             the webpage
├── code/
│   ├── main.py            single-scale alignment (jpgs) and coarse-to-fine pyramid alignment (tifs), SSD metric
│   └── bells_whistles.py  Sobel edge-feature alignment, fixes emir.tif
├── images/
│   ├── course/            the provided data.zip, unzipped (not committed)
│   ├── output/            what the scripts write (not committed)
│   └── results/           the images the webpage shows
└── starter_code/          the course's starter code
```

## Running

```
pip install numpy scikit-image scipy
python code/main.py            # prints each image's G/R (x, y) offset, saves to images/output/
python code/bells_whistles.py  # edge-based alignment of emir.tif -> images/output/out_emir_edges.jpg
```

The scripts find their folders relative to themselves, so they can be run from anywhere.
