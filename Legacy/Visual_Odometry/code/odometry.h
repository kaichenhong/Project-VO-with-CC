#pragma once
#include <time.h>
#include "frame.h"
#include "feature.h"
#include "poseEstimate.h"
#include "triangulate.h"
#include "config.h"
#include "parameters.h"

#define MIN_NUM_FEAT 150		// 175	150(VS)   250(New)

//#define write_Video
//--- Video writer ---//
static cv::VideoWriter writer;
static const int codec = CV_FOURCC('M', 'P', '4', '2');  // select desired codec (must be available at runtime)
static const double fps = 10.0;                          // framerate of the created video stream
static const string filename = "VO_result.avi";			 // name of the output video file
static const cv::Size videoSize = cv::Size(1226, 370);	 // output video file size;		KITTI => (1226, 370)  


/*static string textLine;
static ifstream poseFile(poseLocation);
static double x_pose = 0, y_pose = 0, z_pose = 0;
static double x_prev, y_prev, z_prev;*/

static clock_t loop_start, loop_end;

static const int frameCounter = 1;
static const int firstFrameID = 0;
static double scale = /*0.2*/1.0;

// Vector for tracking time statistics
static vector<float> vTimesTrack;

struct errors 
{
	float r_err;
	float t_err;
	errors(float r_err, float t_err) :
		r_err(r_err), t_err(t_err) {}
};

struct VisualOdometry
{
	vector<Mat> poses_gt;
	vector<Mat> poses_gt_runtime;
	Mat T;
	vector<double> trajDistances;

	vector<Mat> poses_messure;
	Mat T_messure;
	
	vector<errors> err;

	vector<Mat> P_Vec;
	Mat P_Keyframe;

	//vector<Mat> points4D_Vec;
	vector< vector<Point3d> > points3D_Vec;

	Mat R;
	Mat t;

	int ptSizeKF_TH;
	vector<Point2f> ptKF;

	vector<uchar> status;
	vector<uchar> status_after_tracking;
};

bool addFrame(struct Frame &frame);
void odometryRun();
void odometryRunLBP();
void odometryRun_PNP();
void odometryRun_ALL_PNP();
void odometryRun_ALL_PNP2();
void odometryRun_KF();
void odometryRun_PNP_Test();
//void odometryRunLBP_PNP2();

void testingAndCheck();