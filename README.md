
# DARE
This repository contains a python implementation of the Density Adaptive Point Set Registration (DARE) method. Without the density adaptation, the method is equivalent to Joint Registration of Multiple Point Sets (JRMPC) [1]. Additionally, implementations of Color-based Probabilistic Point Set Registration (CPPSR) [2] and Feature-based Probabilistic Point Set Registration (FPPSR) [3] are provided and can be run together with the density adaptation. 

The script reg_demo.py runs DARE on a subsampled version of the vps outdoor dataset. 

## Publication
A detailed description of the DARE method can be found in the CVPR 2018 paper:

F. Järemo Lawin, M. Danelljan, F. S. Khan, P.-E. Forssen, and M. Felsberg, “Density adaptive point set registration,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
<https://arxiv.org/pdf/1804.01495.pdf>

    @InProceedings{jaremo18a,
      author = 	 {Felix J\"aremo Lawin and Martin Danelljan and Fahad Khan and Per-Erik Forss\'en and Michael Felsberg},
      title = 	 {Density Adaptive Point Set Registration},
      booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition},
      year = 	 {2018},
      month = 	 {June},
      address = 	 {Salt Lake City, Utah, USA},
      publisher =    {Computer Vision Foundation},
    }

## Dependencies
* python 3.6
* numpy
* scipy
* matplotlib
* pybind11
* pathlib
* cmake
* pcl (if you want to use FPPSR)

## Installation
Make sure that the above dependencies are installed. 

* Build the pybind module in src/math_utils at src/math_utils/build. 
* To be able to run FPPSR, you need to build the pybind module in src/pcl_utils at src/pcl_utils/build. 

You may use the shell script build_pybind_modules. The code has been tested in Ubuntu 16.04 and 18.04.

## Datasets
The full datasets used in the paper can be found at <http://www.hdrv.org/vps/> and <http://www.prs.igp.ethz.ch/research/completed_projects/automatic_registration_of_point_clouds.html>.

## Contact
Felix Järemo Lawin

email: felix.jaremo-lawin@liu.se

## References
JRMPC:
[1] G. D. Evangelidis, D. Kounades-Bastian, R. Horaud, and E. Z. Psarakis,
“A generative model for the joint registration of multiple point sets,” in
European Conference on Computer Vision, pp. 109–122, Springer, 2014
<https://team.inria.fr/perception/research/jrmpc/>

CPPSR:
[2] M. Danelljan, G. Meneghetti, F. Shahbaz Khan, and M. Felsberg, “A prob-
abilistic framework for color-based point set registration,” in CVPR, 2016.
<http://www.cvl.isy.liu.se/research/cogvis/colored-point-set-registration/index.html>

FPPSR:
[3] M. Danelljan, G. Meneghetti, F. Shahbaz Khan, and M. Felsberg, “Aligning
the dissimilar: A probabilistic method for feature-based point set 
registration,” in ICPR, 2016.
<https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7899641>


