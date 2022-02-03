#pragma once
#include <vector>
#include <numeric>
#include "opencv2\opencv.hpp"
#include "opencv2\xfeatures2d.hpp"
#include "frame.h"
#include "parameters.h"

using namespace std;
using namespace cv;


const float W = 30;
const float HARRIS_K = 0.04f;

void drawOpticalFlow(const Mat &frame, const vector<Point2f> &pt1, const vector<Point2f> &pt2);

void detectKeypoints(const Mat &img, vector<Point2f> &pt);

bool trackKeypoints(Mat &img1, Mat &img2, vector<Point2f> &pt1, vector<Point2f> &pt2, const int &keyFramePtSizeTH);
bool trackKeypoints(Mat &img1, Mat &img2, vector<Point2f> &pts1, vector<Point2f> &pts2, vector<uchar> &status, const int &keyFramePtSizeTH);
bool trackKeypoints(Mat &img1, Mat &img2, vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, const int &keyFramePtSizeTH);
bool trackKeypoints(Mat &img1, Mat &img2, vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, const int &keyFramePtSizeTH, vector< vector<Point3d> > &points3D, const bool &have_3DPt);
void trackKeypoints(struct Frame &_frame1, struct Frame &_frame2, vector<Point2f> &ptKF, const int &keyFramePtSizeTH, vector< vector<Point3d> > &points3D_Vec, const bool &have_3DPt);
