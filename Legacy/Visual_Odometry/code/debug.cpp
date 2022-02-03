#include "debug.h"

void 
debug() 
{
	int frameID_1[2] = { 551, 552 };
	int frameID_2[2] = { 65, 66 };
	int frameID_3[2] = { 65, 67 };
	int *testFrameID = frameID_3;
	clock_t t1_1, t1_2;

	struct Frame lastFrame;
	int frameID = 551;
	getFrame(lastFrame, testFrameID[0]);
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);
	int keyFrameSize = lastFrame.pt.size();


	struct Frame currentFrame;
	int frameID2 = 552;
	getFrame(currentFrame, testFrameID[1]);
	Mat currentFrameLBP;
	t1_1 = clock();
	Mat T;


	Mat show;
	vector<KeyPoint> keypoints;
	//KeyPoint::convert(lastFrame.pt, keypoints);
	drawKeypoints(currentFrame.img, keypoints, show, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
	//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
	imshow("test", show);
	//imwrite("test.jpg", show);

	waitKey(0);
}