#pragma once
#include "opencv2\opencv.hpp"

using namespace cv;


#define Use_datasets_01
//#define Use_datasets_02
//#define Use_datasets_03

//#define Use_datasets_Dynamic_Scene

//#define Use_ZED_Camera


#ifdef Use_datasets_01
const double focal = 707.0912;
const Point2d pp(601.8873, 183.1104);
const Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);

const int ROW = 720, COL = 720;

const char fileLocation[] = "D:/Document/The KITTI Vision Benchmark Suite/dataset/sequences/04/image_0/%06d.png";
const char poseLocation[] = "D:/Document/The KITTI Vision Benchmark Suite/dataset/poses/04.txt";

//-- parameters --//
// for feature.cpp
const int nfeaturesCell = 4;

// for poseEstimate
const float maxDistanceErr_F = 0.2f;
const float poseCorrectRatio = 0.5f;
#endif

#ifdef Use_datasets_02
const double focal = 707.0912;
const Point2d pp(601.8873, 183.1104);
const Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);

const int ROW = 720, COL = 720;

const char fileLocation[] = "D:/Document/The KITTI Vision Benchmark Suite/dataset/sequences/06/image_0/%06d.png";
const char poseLocation[] = "D:/Document/The KITTI Vision Benchmark Suite/dataset/poses/06.txt";

//-- parameters --//
// for feature.cpp
const int nfeaturesCell = 4;

// for poseEstimate
const float maxDistanceErr_F = 0.2f;
const float poseCorrectRatio = 0.5f;
#endif

#ifdef Use_datasets_Dynamic_Scene
const double focal = 707.0912;
const Point2d pp(601.8873, 183.1104);
const Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);

const int ROW = 720, COL = 720;

const char fileLocation[] = "D:/Document/The KITTI Vision Benchmark Suite/dataset/sequences/07/image_0/%06d.png";
const char poseLocation[] = "D:/Document/The KITTI Vision Benchmark Suite/dataset/poses/07.txt";

//-- parameters --//
// for feature.cpp
const int nfeaturesCell = 13;

// for poseEstimate
const float maxDistanceErr_F = 0.5f;
const float poseCorrectRatio = 0.4f;
#endif

#ifdef Use_datasets_03
const double focal = 707.0912;
const Point2d pp(601.8873, 183.1104);
const Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);

const int ROW = 720, COL = 720;

const char fileLocation[] = "D:/Document/The KITTI Vision Benchmark Suite/dataset/sequences/09/image_0/%06d.png";
const char poseLocation[] = "D:/Document/The KITTI Vision Benchmark Suite/dataset/poses/09.txt";

//-- parameters --//
#define use_flag
// for feature.cpp
const int nfeaturesCell = 4;

// for poseEstimate
const float maxDistanceErr_F = 0.3f;
const float poseCorrectRatio = 0.4f;
#endif

#ifdef Use_ZED_Camera
const double focal = 699.845;
const Point2d pp(682.715, 344.054);
const Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);

const int ROW = 579, COL = 603;		// NCTU Map size

const char fileLocation[] = "D:/Document/NCTU/MOST/image_00/%06d.jpg";

// for compile pass. Don't use.
const char poseLocation[] = "D:/Document/The KITTI Vision Benchmark Suite/dataset/poses/00.txt";
#endif