#pragma once
#include "config.h"

//////////////////////////////////////////////////
//               keyframe (abandon)             //
//////////////////////////////////////////////////
#define pts_Size_KF_Ratio 0.8f
#define pts_Size_KF_TH 100

//////////////////////////////////////////////////
//                    feature                   //
//////////////////////////////////////////////////
#define edgeThreshold 31		// ORB => 31	11(New) 11(KF)  16(KFnew)

// in a cell(bucket)...
#define maxCorners 0			// 0 implies that no limit on the maximum.	Make test data => 100.
#define qualityLevel 0.01
#define minDistance 10
#define detectBlockSize 3

// optical flow tracking...
#define windowSize 17			// CV=>21  15	7(VS)   7		5(New)  9(KF)   11 7 15(KFnew)  17(口試論文本)
#define maxLevel 30		        // maximal pyramid level number; If set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on;  CV=>3	  4		7(New)  9(KF) 9(KFnew)
#define klt_max_iter 10000		// => maximum  CV=>30	20   10      15		10(New) 35(KF)  50(KFnew)

// keyframe decision...
#define keyFramePtSizeThreshold 0.5   // 0.8(KF)(ORB_10000 pts)  0.5(KFnew)

//////////////////////////////////////////////////
//               poseEstimate                   //
//////////////////////////////////////////////////

// homography matrix
#define maxDistanceErr_H 1.0f

// fundamental matrix
//#define maxDistanceErr_F 0.2f	// 0.5f(口試論文本)(KITTI_07)	0.3(KITTI_09)	0.2(KITTI_06 / 04)

// pose correct ratio
//#define poseCorrectRatio 0.5f	// 0.35(New)  0.6(KF)  0.4(KITTI_07 / 09) or 0.5(KFnew)(口試論文本)

// solvepnp parameters
#define iterationsCount 100000	// => maximum
#define reprojectionError 0.6f	// 1.0(口試論文本 / KITTI_07(lost para) )	0.6(seems better 04 06 07 09)