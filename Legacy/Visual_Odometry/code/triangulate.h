#pragma once
#include <vector>
#include "opencv2\opencv.hpp"
#include "config.h"

using namespace std;
using namespace cv;

void normalizePoints(const Mat &_points4D, vector<Point3d> &_points3D);

void
showTriangulation(
	const vector<Point2f> &keypoint_1,
	const vector<Point2f> &keypoint_2,
	const Mat &R, const Mat &t,
	const vector<Point3d> &points);

void 
triangulation( 
    const vector<Point2f> &keypoint_1,
    const vector<Point2f> &keypoint_2,
    const Mat &R, const Mat &t, 
    vector<Point3d> &points);

void
triangulate_Points(const Mat &P1, const Mat &P2,  const vector<Point2f> &pts1, const vector<Point2f> &pts2, vector<uchar> &status, vector<Point3d> &pts3D);