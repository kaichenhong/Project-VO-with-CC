#pragma once
#include <vector>
#include <opencv2\opencv.hpp>
#include "config.h"
#include "parameters.h"

using namespace std;

#define sigma 1.0				// for CheckF & CheckH

void poseInitial(Mat &R, Mat &t);
bool poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, Mat &R_f, Mat &t_f, const bool keyFrame);
bool poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, Mat &T, const bool keyFrame);
bool poseEstimate(vector<Point2f> &pts1, vector<Point2f> &pts2, vector<uchar> &status, Mat &T, const bool keyFrame);
bool poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, Mat &T, const bool keyFrame);
bool poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, Mat &T);
bool poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, vector< vector<Point3d> > &pt3D_Vec, Mat &T, const bool &have_3DPt);
bool poseEstimate(vector<Point2f> &_pt1, vector<Point2f> &_pt2, Mat &_T);
Mat poseEstimatePnP(const vector<Point3d> &pts3D, const vector<Point2f> &pts, const vector<uchar> &status);