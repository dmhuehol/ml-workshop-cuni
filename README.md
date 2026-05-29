# ml-workshop-cuni
Code for machine learning (ML) workshop at the Charles University Katedra fyziky atmosféry (May 2026) with different demonstration applications of neural networks implemented in Tensorflow in Python.

### Table of contents
* [1. `fit_sinewave`](#1-fit_sinewave)
* [2. `nn_class_palmerpenguins`](#2-nn_class_palmerpenguins)  
* [3. `ann_ozone_joshuatree`](#3-ann_ozone_joshuatree)
* [4. notes.md](#4-notesmd)
* [5. Workshop information and slides](#5-workshop-information-and-slides)  

## 1. `fit_sinewave`
Fit a sine wave using an artificial neural network. This provides the easiest "quick start" to playing around with fundamentals of ANNs!

#### 1.1. License information for `fit_sinewave`
`fit_sinewave` is available under an MIT License (c) [Elizabeth Barnes](barnes-research.com) 2022. See the license file `LICENSE_fitsinewave-ann_ozone_joshuatree` in this repository. It has been lightly updated by [Daniel Hueholt](https://hueholt.earth) from the original version used in Arcodia et al. 2022 "Applied Machine Learning Tutorial for Earth Scientists": [Zenodo link to original](https://doi.org/10.5281/zenodo.6686878). This code originally written by [Emily Gordon](https://emilymgordon.com) and [Frances Davenport](https://fdavenport.github.io).

## 2. `nn_class_palmerpenguins`
Classify penguins from the Palmer Penguins dataset using an artificial neural network. This provides an example of a classification problem using a real-world dataset.

#### 2.1. License information for `nn_class_palmerpenguins`
`nn_class_palmerpenguins` is available under an MIT License (c)[Daniel Hueholt](https://hueholt.earth) 2025. See the license file `LICENSE_nn_class_palmerpenguins` in this repository. It has been lightly updated from an original version written by Daniel Hueholt: [GitHub link to original](https://github.com/dmhuehol/palmerpenguins-classifiers).

## 3. `ann_ozone_joshuatree`
Predict ozone concentrations (regression problem) at Joshua Tree National Park from meteorological variables using an explainable neural network. This is a more complex task than the other two and additionally demonstrates incorporating domain knowledge of the data. This notebook uses DeepSHAP to provide an explanation for the model's performance.

#### 3.1. License information for `ann_ozone_joshuatree`
`ann_ozone_joshuatree` is available under an MIT License (c) [Elizabeth Barnes](barnes-research.com) 2022. See the license file `LICENSE_fitsinewave-ann_ozone_joshuatree` in this repository. It has been lightly updated by [Daniel Hueholt](https://hueholt.earth) from the original version written by TA [Jamin Rader](https://jaminrader.com) for ATS 780A7 Spring 2022 at Colorado State University led by Prof. Elizabeth Barnes: [GitHub link to original](https://github.com/eabarnes1010/course_ml_ats/tree/main).

## 4. notes.md
notes.md provides notes on each notebook including relevant hyperparameters, tuning information, and further information about what each notebook demonstrates. These are kept separate from the main notebooks so that a new user can have the ideal "spoiler-free" experience!

## 5. Workshop information and slides
Workshop led by [Daniel Hueholt](https://hueholt.earth), who also maintains this repository.  

Slides from the research seminar (May 19), lecture on climate data science (May 20), and workshop on ML applications (May 26-27) are accessible through Google Drive: [link](https://drive.google.com/drive/folders/1jSVGBowETxwXdD1yYBwDwIXDVpI9d2aD?usp=sharing).
