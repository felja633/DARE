#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

Eigen::MatrixXf mat_exp(Eigen::MatrixXf mat)
{
  return mat.array().exp();
} 

Eigen::MatrixXf mat_exp_par(Eigen::MatrixXf mat)
{
  #pragma omp parallel for
  for(unsigned int c = 0; c < mat.cols(); ++c) {
    
    for(unsigned int r = 0; r < mat.rows(); ++r)
    {
      mat(r,c) = std::exp(mat(r,c));
    }
  }
  return mat;
}

Eigen::MatrixXf mat_mult(Eigen::MatrixXf A, Eigen::MatrixXf B)
{
  return A*B;
}

Eigen::MatrixXf point_mult(Eigen::MatrixXf A, Eigen::MatrixXf B)
{
  return A.cwiseProduct(B);
}

Eigen::MatrixXf point_mult_par(Eigen::MatrixXf A, Eigen::MatrixXf B)
{
  #pragma omp parallel for
  for(unsigned int c = 0; c < A.cols(); ++c) {
    
    for(unsigned int r = 0; r < A.rows(); ++r)
    {
      A(r,c) *= B(r,c);
    }
  }
  return A;
} 

Eigen::MatrixXf mat_sqe(Eigen::MatrixXf Y, Eigen::MatrixXf X)
{
  //return mat.array().exp();
  float x,y,z;
  float diffx,diffy,diffz;
  Eigen::MatrixXf sqe = Eigen::MatrixXf::Zero(Y.rows(), X.rows());
  #pragma omp parallel for
  for(unsigned int r = 0; r < Y.rows(); ++r)
  {
    x = Y(r,0); y = Y(r,1); z = Y(r,2);
    for(unsigned int rx = 0; rx < X.rows(); ++rx)
    {
      diffx = x-X(rx,0); diffy = y-X(rx,1); diffz = z-X(rx,2);
      sqe(r, rx) = diffx*diffx+diffy*diffy+diffz*diffz;
    }
  }

  return sqe;
} 


PYBIND11_MODULE(math_utils, m)
{
  m.def("mat_exp", &mat_exp);
  m.def("mat_exp_par", &mat_exp_par);
  m.def("mat_sqe", &mat_sqe);
  m.def("mat_mult", &mat_mult);
  m.def("point_mult", &point_mult);
  m.def("point_mult_par", &point_mult_par);
}
