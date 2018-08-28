#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <iostream>

namespace py = pybind11;

class PCLUtilsClass {

  public:
  PCLUtilsClass() {is_set_normals_=false;}
  //~PCLUtils() {}
  
  Eigen::MatrixXd matrix_add(Eigen::MatrixXd A, Eigen::MatrixXd B)
  {
    return A+B;
  }

  void compute_normals(Eigen::MatrixXd points, unsigned int num_neighbors=10)
  {
    int rows = points.rows();
    int cols = points.cols();
  
    //initialize with sometihng
    normals_eigen_ =  Eigen::MatrixXd::Zero(rows, cols);

    //NdArray<float> M(scalars, shape, strides, ndim);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>(cols, 1));
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  
    // Copy points from input matrix
    for (size_t i = 0; i < cols; i++) {
        pc->points[i].x = (float)points(0, i);
        pc->points[i].y = (float)points(1, i);
        pc->points[i].z = (float)points(2, i);
        //fprintf(stderr, "x = %f, y = %f, z = %f\n", pc->points[i].x, pc->points[i].y, pc->points[i].z);
    }

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(pc);
    ne.setSearchMethod(tree);
    ne.setKSearch(num_neighbors);
    ne.compute(*cloud_normals);

    // Copy result to input matrix
    for (size_t i = 0; i < cols; i++) {
        normals_eigen_(0, i) = (double)cloud_normals->points[i].normal_x;
        normals_eigen_(1, i) = (double)cloud_normals->points[i].normal_y;
        normals_eigen_(2, i) = (double)cloud_normals->points[i].normal_z;
    }

    //fprintf(stderr, "copying pcl normals...\n");
    //pcl::copyPointCloud(*cloud_normals, *normals_);
    is_set_normals_ = true;

  }

  void estimate_fpfh_descriptors(Eigen::MatrixXd points, unsigned int num_neighbors)
  {
    // Verify the arguments

    int num_bins = 33;

    int rows = points.rows();
    int cols = points.cols();

    pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>(cols, 1));
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    // Copy points from input matrix
    for (size_t i = 0; i < cols; i++) {
        pc->points[i].x = (float)points(0, i);
        pc->points[i].y = (float)points(1, i);
        pc->points[i].z = (float)points(2, i);
        //fprintf(stderr, "x = %f, y = %f, z = %f\n", pcl->points[i].x, pcl->points[i].y, pcl->points[i].z);
    }

    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>(cols,1));

    if (is_set_normals_ == false) {
        compute_normals(points, 10);

    }

    for (size_t i = 0; i < cols; i++) {
        normals->points[i].normal_x = (float)normals_eigen_(0, i);
        normals->points[i].normal_y = (float)normals_eigen_(1, i);
        normals->points[i].normal_z = (float)normals_eigen_(2, i);
    }
    // Run the estimator

    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(pc);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);
    fpfh.setKSearch(num_neighbors);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descs(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh.compute(*descs);

    // Copy result from PCL to input matrix
    fpfh_features_eigen_ = Eigen::MatrixXd::Zero(num_bins, cols);
    for (size_t i = 0; i < cols; i++) {
        for (size_t j = 0; j < num_bins; j++) {
            //std::cout<<"(double)descs->points[i].histogram[j] = "<<(double)descs->points[i].histogram[j]<<std::endl;
            fpfh_features_eigen_(j, i) =(double)descs->points[i].histogram[j];
        }
    }
    descs.reset();
  }

  Eigen::MatrixXd get_normals() {return normals_eigen_;}
  Eigen::MatrixXd get_fpfh_features() {return fpfh_features_eigen_;}
  private:
    bool is_set_normals_;
    Eigen::MatrixXd normals_eigen_;
    Eigen::MatrixXd fpfh_features_eigen_;
  

};


/*int dummy_func(int x) {
  return x*x;
}*/

// ----------------
// Python interface
// ----------------

PYBIND11_MODULE(pcl_utils, m)
{

  py::class_<PCLUtilsClass>(m, "PCLUtilsClass")
      .def(py::init<>())
      .def("matrix_add", &PCLUtilsClass::matrix_add)
      .def("compute_normals", &PCLUtilsClass::compute_normals)
      .def("estimate_fpfh_descriptors", &PCLUtilsClass::estimate_fpfh_descriptors)
      .def("get_normals", &PCLUtilsClass::get_normals)
      .def("get_fpfh_features", &PCLUtilsClass::get_fpfh_features);

//m.def("dummy_func", &dummy_func);
}
