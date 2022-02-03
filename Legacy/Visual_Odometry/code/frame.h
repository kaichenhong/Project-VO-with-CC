#pragma once
#include <vector>
#include "opencv2\opencv.hpp"
#include "config.h"

using namespace std;
using namespace cv;


struct Frame {
    int id;

    Mat img;
    Mat grayImg;
	vector<Mat> mvImagePyramid;

    vector<Point2f> pt;
	vector<KeyPoint> keypoints;
	vector<vector<KeyPoint> > allKeypoints;
	Mat descriptors;

	Mat T;
	Mat P;
    Mat R_f;
    Mat t_f;
};

struct Frame *getFrame(int frameNum);
bool getFrame(struct Frame &frame, int frameNum);