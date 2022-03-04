# Method for running experiments in Part B

## Using focal loss
Use (uncomment) the following line from loss_function.py:
```
heatmap_loss = heatmap_focal_loss(target_heatmap, predicted_heatmap, gamma=gamma, alpha=alpha)
```
And comment out:
```
heatmap_loss = ((target_heatmap - predicted_heatmap) ** 2).mean()
```
Tuning hyperparameters alpha and gamma e.g. :
```
gamma = 3.0
alpha = 0.25
```

## Creating general heatmap using rotated anisotropic gaussian kernal
Use (uncomment) the following line from loss_target.py:
```
heatmap = create_general_heatmap(grid_coords, center=center, scale_x=scale_x, scale_y=scale_y, headings=torch.tensor([math.sin(yaw), math.cos(yaw)]))  # [H x W]
```

And comment out:
```
heatmap = create_heatmap(grid_coords, center=center, scale=scale)  # [H x W
```

Standard scale_x and scale_y used:
```
scale_x = 2*(x_size**2) / self._heatmap_norm_scale
scale_y = 2*(y_size**2)/ self._heatmap_norm_scale
```

Tuning parameters scale_x and scale_y e.g.:
```
scale_x = 1*(x_size**2) / self._heatmap_norm_scale
scale_y = 1*(y_size**2)/ self._heatmap_norm_scale
```

## Using focal loss and rotated anisotropic gaussian kernal
Use the combination of the commenting and uncommenting outlined above to modify code.





# CSC490H1: Making Your Self-driving Car Perceive the World

This repository contains the starter code for CSC490H1:
Making Your Self-driving Car Perceive the World.

## Getting started

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

   ```bash
   curl 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh' > Miniconda.sh
   bash Miniconda.sh
   rm Miniconda.sh
   ```

2. Close and re-open your terminal session.

3. Change directories (`cd`) to where you cloned this repository.

4. Create a new conda environment:

   ```bash
   conda env create --file environment.yml
   ```

5. Activate your new environment:

   ```bash
   conda activate csc490
   ```

6. Download [PandaSet](https://scale.com/resources/download/pandaset).
   After submitting your request to download the dataset, you will receive an
   email from Scale AI with instructions to download PandaSet in three parts.
   Download Part 1 only. After you have downloaded `pandaset_0.zip`,
   unzip the dataset as follows:

   ```bash
   unzip pandaset_0.zip -d <your_path_to_dataset>
   ```
