# Stereo Scene Reconstruction from 2D Images

**Computer Vision | Academic Project | Heinrich Heine University Düsseldorf**

## Project Overview

This project explores the reconstruction of a 3D scene from a pair of stereo images using classical computer vision techniques. The main objective is to estimate depth information from two images captured from different viewpoints and investigate methods for reconstructing the underlying scene geometry. The project covers stereo geometry, image rectification, disparity estimation, depth estimation, graphical-model-based smoothing, and 3D surface extraction.

## Technologies and Libraries

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Numerical Computing:** NumPy
- **Image Processing:** Pillow
- **Visualization:** Matplotlib
- **Additional Tools:** tqdm, gco (where available)
- **Development Environment:** Google Colab, Jupyter Notebook

## Methodology

### 1. Fundamental Matrix Estimation and Image Rectification

The first stage investigates the geometric relationship between two stereo images.
Key tasks include:
- Loading and visualizing stereo image pairs.
- Estimating the fundamental matrix.
- Investigating epipolar geometry.
- Rectifying stereo images to facilitate correspondence matching.

### 2. Disparity and Depth Estimation

The next stage explores disparity estimation between corresponding image points. Disparity represents the difference in the image positions of corresponding scene points. This information can be used to estimate depth when the required camera calibration parameters are available.

The workflow includes:
- Identifying corresponding image regions.
- Estimating disparity values.
- Generating and visualizing disparity maps.
- Investigating the relationship between disparity and depth.

### 3. Disparity Map Smoothing

The project investigates graphical-model-based methods for improving estimated disparity maps. The objective is to reduce noisy or inconsistent disparity estimates while preserving meaningful scene boundaries.

### 4. 3D Surface Reconstruction

The final stage explores reconstructing the visible scene surface from estimated depth information. The workflow focuses on transforming image-based depth information into a representation of the underlying 3D scene.

## Repository Structure

```text
stereo-scene-reconstruction/
├── README.md
├── stereo_scene_reconstruction.ipynb
└── functions_student.py
```

**stereo_scene_reconstruction.ipynb:** Main notebook containing the project's experiments, processing stages, and visualizations.

**functions_student.py:** Supporting Python functions used throughout the reconstruction workflow.

## Dataset

The project uses stereo image data from the **2014 Stereo Dataset**, referenced in the university assignment. The repository does not include the image dataset. Dataset access, usage, and redistribution should follow the original provider's licence and conditions.

## Running the Project

The project was developed using Google Colab.
1. Open `stereo_scene_reconstruction.ipynb` in Google Colab.
2. Install or make available the required Python dependencies.
3. Upload `functions_student.py` to the notebook's working directory.
4. Obtain the required stereo images and place them in the directory structure expected by the notebook.
5. Execute the notebook cells sequentially.

The complete workflow has not yet been independently tested in a clean environment.

## Results and Evaluation

The notebook contains the experiments and visualizations associated with the stereo reconstruction workflow. Quantitative results and reconstruction quality have not yet been independently verified for this README.

## Academic Context and Acknowledgements

This project was developed as part of Computer Vision coursework in the M.Sc. Artificial Intelligence and Data Science programme at Heinrich Heine University Düsseldorf. The work is based on a university-provided assignment involving stereo scene reconstruction. The assignment specifies group work, and individual contributions and university-provided code should be credited appropriately.

GitHub: [MdRasel-Hossain](https://github.com/MdRasel-Hossain)
