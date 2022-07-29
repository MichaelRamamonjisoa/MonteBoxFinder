# Cuboid Detection

## Install 

### Dependencies

#### CGAL
Our work uses [CGAL](https://doc.cgal.org/latest/Manual/installation.html), which we download and link to our executable. 
```
wget https://github.com/CGAL/cgal/releases/download/v5.2.2/CGAL-5.2.2.tar.xz
tar xf CGAL-5.2.2.tar.xz
```

#### Eigen
Install Eigen3 using:
```
sudo apt-get install libeigen3-dev
```

#### OpenMP

Install OpenMP using:
```
sudo apt-get install libomp-dev
```

### Run installation
Run [install.py](https://github.com/MichaelRamamonjisoa/MonteBoxFinder/tree/main/CuboidDetection/install.py):
```
python install.py
```

## Run Cuboid Detection
Run the cuboid detection script [run_cuboid_detector.py](https://github.com/MichaelRamamonjisoa/MonteBoxFinder/blob/main/python/run_cuboid_detector.py) using 
```
python run_cuboid_detector.py --scans_dir PATH_TO_SCANNET_SCENES --out_dir ../Data/PrimitiveDetection --lib_dir ../CuboidDetection/build
```

