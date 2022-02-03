#define _CRT_SECURE_NO_WARNINGS
#include "odometry.h"

static int frameNum = 0;

static void
remove_bad_points(vector<Point2f> &pts, const vector<uchar> &status)
{
	size_t i, k = 0;
	for (i = k = 0; i < pts.size(); i++) {
		if (!status[i]) {
			continue;
		}
		pts[k] = pts[i];
		k = k + 1;
	}

	//int removed_points = pts.size() - k;

	pts.resize(k);
}

static void
remove_bad_points(vector< vector<Point3d> > &pts3D_vec, const vector<uchar> &status)
{
	const unsigned int N = pts3D_vec.size();
	const unsigned int M = pts3D_vec[0].size();
	size_t i, k = 0, m;

	for (m = 0; m < N; m++) {
		for (i = k = 0; i < M; i++) {
			if (!status[i]) {
				continue;
			}
			pts3D_vec[m].at(k) = pts3D_vec[m].at(i);
			k = k + 1;
		}
		//int removed_points = pts.size() - k;

		pts3D_vec[m].resize(k);
	}
}

inline static void
T2Rt(const Mat &T, Mat &R, Mat &t)
{
	R = ( T( Range(0, 3), Range(0, 3) ) ).clone();
	t = ( T( Range(0, 3), Range(3, 4) ) ).clone();
}

inline static void
T2t(const Mat &T, Mat &t)
{
	t = ( T( Range(0, 3), Range(3, 4) ) ).clone();
}

static double
normOfTransform(const Mat &R, const Mat &t)
{
#define M_PI (3.14159265358979323846)
	//cout << "R" << rvec << endl;
	//cout << "t" << tvec << endl;
	Mat rvec, tvec;
	Rodrigues(R, rvec);
	Rodrigues(t, tvec);
	//cout << "Norm: R: " << cv::norm(rvec) << "   2PI-R: " << 2 * M_PI - cv::norm(rvec) << "   t: " << cv::norm(tvec) << endl;
	return fabs(min(cv::norm(rvec), 2 * M_PI - cv::norm(rvec))) + fabs(cv::norm(tvec));
}

inline static float 
rotationError(Mat &_pose_error) {
	float a = _pose_error.at<double>(0, 0);
	float b = _pose_error.at<double>(1, 1);
	float c = _pose_error.at<double>(2, 2);
	float d = 0.5*(a + b + c - 1.0);
	return acos(max(min(d, 1.0f), -1.0f));
}

inline static float 
translationError(Mat &_pose_error) {
	float dx = _pose_error.at<double>(0, 3);
	float dy = _pose_error.at<double>(1, 3);
	float dz = _pose_error.at<double>(2, 3);
	return sqrt(dx*dx + dy*dy + dz*dz);
}

inline static void
updataPose(const double scale, struct VisualOdometry &vo, const Mat &_T)
{
	Mat R, t;
	T2Rt(_T, R, t);
	t *= scale;

	Mat T = Mat::eye(4, 4, CV_64F);
	R.copyTo(T(Range(0, 3), Range(0, 3)));
	t.copyTo(T(Range(0, 3), Range(3, 4)));

	vo.T_messure *= T;
}

inline static void
updataPose(const double _scale, const Mat &_T_f, Mat &_T_messure)
{
	Mat R, t;
	T2Rt(_T_f, R, t);
	t *= _scale;

	Mat T = Mat::eye(4, 4, CV_64F);
	R.copyTo(T(Range(0, 3), Range(0, 3)));
	t.copyTo(T(Range(0, 3), Range(3, 4)));

	_T_messure *= T;
}

inline static Mat 
inverseTransformation(const Mat &_rvec, const Mat &_tvec) 
{
	Mat R, tvec;
	Rodrigues(_rvec, R);

	R = R.t();           // inverse rotation
	tvec = -R * _tvec;   // translation of inverse

	// camPose is a 4x4 matrix with the pose of the camera in the object frame
	Mat T = Mat::eye(4, 4, CV_64F);
	R.copyTo(T.rowRange(0, 3).colRange(0, 3));		// copies R into camPose
	tvec.copyTo(T.rowRange(0, 3).colRange(3, 4));	// copies tvec into camPose

	return T;
}

inline static double
getAbsoluteScale(const Mat &T)
{
	Mat t;
	static Mat t_prev = Mat::zeros(3, 1, CV_64FC1);

	T2t(T, t);
	double scale = norm(t, t_prev);
	t_prev = t.clone();

	return scale;
}

inline static Mat 
makeProjectionMatrix(const Mat &_T) 
{
	Mat R = _T(Range(0, 3), Range(0, 3));
	Mat t = _T(Range(0, 3), Range(3, 4));

	Mat P(3, 4, CV_64F);

	P(Range(0, 3), Range(0, 3)) = R.t();
	P(Range(0, 3), Range(3, 4)) = -R.t()*t;
	P = K * P;

	return P;
}

static void 
showFrame(const struct Frame &lastFrame, const struct Frame &currentFrame)
{
	Mat show;
	vector<KeyPoint> keypoints;

	KeyPoint::convert(lastFrame.pt, keypoints);
	drawKeypoints(currentFrame.img, keypoints, show, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);

	putText(show, "FPS: " + to_string((1 / ((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)))), Point(5, lastFrame.img.rows - 40), 1, 1, Scalar(0, 255, 0), 2);

	imshow("frame", show);
}

/*inline static void
showTraj(Mat &traj, const Mat &t)
{
	circle(traj, Point(t.at<double>(0) + COL / 2, COL - (t.at<double>(2) + ROW / 4)), 1, CV_RGB(255, 0, 0), 2);

	imshow("Trajectory", traj);
}*/

inline static void
showTrajAndGTruth(Mat &traj, const Mat &T_messure, const Mat &T)
{
	Mat t_messure;
	T2t(T_messure, t_messure);
	circle(traj, Point(t_messure.at<double>(0) + COL / 2, COL - (t_messure.at<double>(2) + ROW / 4)), 1, CV_RGB(255, 0, 0), 2);

	Mat t_GT;
	T2t(T, t_GT);
	circle(traj, Point(t_GT.at<double>(0) + COL / 2, COL - (t_GT.at<double>(2) + ROW / 4)), 1, CV_RGB(0, 200, 0), 2);

	imshow("Trajectory", traj);
}

inline static void 
createGrid(vector<Point2f> &_grid, const int &_wRes, const int &_hRes, const int &_wStep, const int &_hStep) 
{
	//_grid.clear();
	for (int i = edgeThreshold; i < _wRes - edgeThreshold; i += _wStep)
		for (int j = edgeThreshold; j < _hRes - edgeThreshold; j += _hStep)
			_grid.push_back(Point2f(i, j));
}

/*static void 
showTraj(Mat &traj, const Mat &t) 
{	
	// draw trajectory
	int x = int(t.at<double>(0)) + COL / 2;
	int y = int(t.at<double>(2)) + ROW / 4;
	//printf("%d %d \n", x, y);

	circle(traj, Point(x, COL - y), 1, CV_RGB(255, 0, 0), 2);
	//circle(traj, Point( t.at<double>(0) + COL / 2, COL - (t.at<double>(2) + ROW / 4) ), 1, CV_RGB(255, 0, 0), 2);

	circle(traj, Point(x_pose + COL / 2, COL - (z_pose + ROW / 4)), 1, CV_RGB(0, 200, 0), 2);

	//rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);

	//char text[100];
	//cv::Point textOrg(10, 50);
	//int fontFace = FONT_HERSHEY_PLAIN;
	//double fontScale = 1;
	//int thickness = 1;
	//sprintf(text, "x=%.02f(%.02f)m y=%.02f(%.02f)m z=%.02f(%.02f)m", t.at<double>(0), x_pose, t.at<double>(1), y_pose, t.at<double>(2), z_pose);
	//putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
	
	imshow("Trajectory", traj);
}*/

/*static double 
calcAbsoluteScale()
{
	z_prev = z_pose;
	x_prev = x_pose;
	y_prev = y_pose;
	std::istringstream in(textLine);
	//cout << textLine << '\n';
	for (int j = 0; j<12; j++) {
		in >> z_pose;
		if (j == 7) y_pose = z_pose;
		if (j == 3) x_pose = z_pose;
	}
	double scale = sqrt((x_pose - x_prev)*(x_pose - x_prev) + (y_pose - y_prev)*(y_pose - y_prev) + (z_pose - z_prev)*(z_pose - z_prev));
	//cout << "Scale is " << scale << endl;
	return scale;
}*/

static void 
getAbsoluteTransformationMat(const int &frameID, Mat &T)
{
	int i = 0;
	string line;
	vector<double> matrixNum;

	ifstream poseFile(poseLocation);
	if ( poseFile.is_open() ) {
		while ( ( getline(poseFile, line)) && ( i <= frameID) ) {
			istringstream in(line);
			//cout << line << endl << endl;
			if (i == frameID) {
				for (int j = 0; j < 12; j++) {
					double num = 0;
					in >> num;
					matrixNum.push_back(num);
				}
				T = ( Mat_<double>(4, 4) << matrixNum.at(0), matrixNum.at(1), matrixNum.at(2), matrixNum.at(3), 
										    matrixNum.at(4), matrixNum.at(5), matrixNum.at(6), matrixNum.at(7), 
										    matrixNum.at(8), matrixNum.at(9), matrixNum.at(10), matrixNum.at(11),
														  0,			   0,				 0,				   1 );
			}
			i++;
		}
		poseFile.close();
	}
	else {
		printf("can't open the pose file !!! \n \n");
	}
}

static void 
loadPoses(vector<Mat> &_poses) 
{
	string line;
	vector<double> matrixNum;
	ifstream poseFile(poseLocation);

	if (poseFile.is_open()) {
		while (getline(poseFile, line)) {
			istringstream in(line);
			//cout << line << endl << endl;
			for (int j = 0; j < 12; j++) {
				double num = 0;
				in >> num;
				matrixNum.push_back(num);
			}
			Mat T = (Mat_<double>(4, 4) << matrixNum.at(0), matrixNum.at(1),  matrixNum.at(2),  matrixNum.at(3),
										   matrixNum.at(4), matrixNum.at(5),  matrixNum.at(6),  matrixNum.at(7),
										   matrixNum.at(8), matrixNum.at(9), matrixNum.at(10), matrixNum.at(11),
														 0,				  0,				0,				 1);
			_poses.push_back(T.clone());
			matrixNum.clear();
		}
		poseFile.close();
	}
	else {
		printf("can't open the pose file !!! \n \n");
	}
}

static void 
trajectoryDistances(const vector<Mat> &_poses, vector<double> &_dist) {
	_dist.push_back(0);

	for (int32_t i = 1; i < _poses.size(); i++) {
		Mat T1 = _poses[i - 1];
		Mat T2 = _poses[i];
		float dx = T1.at<double>(0, 3) - T2.at<double>(0, 3);
		float dy = T1.at<double>(1, 3) - T2.at<double>(1, 3);
		float dz = T1.at<double>(2, 3) - T2.at<double>(2, 3);
		_dist.push_back(_dist[i - 1] + sqrt(dx*dx + dy*dy + dz*dz));
	}
}

static void 
CalcSequenceErrors(const vector<Mat> &_poses_gt, const vector<Mat> &_poses_messure, vector<errors> &_err, const int _kfID, const int _currID, const double _len) 
{
	int size_pose_arr = _poses_messure.size() - 1;
	//Mat pose_delta_gt = _poses_gt[size_pose_arr - 1].inv() * _poses_gt[size_pose_arr];
	Mat pose_delta_gt = _poses_gt[_kfID].inv() * _poses_gt[_currID];
	Mat pose_delta_messure = _poses_messure[size_pose_arr - 1].inv() * _poses_messure[size_pose_arr];
	Mat pose_error = pose_delta_messure.inv() * pose_delta_gt;
	
	float r_err = rotationError(pose_error);
	float t_err = translationError(pose_error);

	_err.push_back(errors(r_err/*/_len*/, t_err/*/_len*/));
}

static void
poseFrameErrors(const vector<Mat> &_poses_gt, const Mat &_T_f, const int _lastID, const int _currID, vector<errors> &_err)
{
	Mat pose_delta_gt = _poses_gt[_lastID].inv() * _poses_gt[_currID];
	Mat pose_error = _T_f.inv() * pose_delta_gt;

	float r_err = rotationError(pose_error);
	float t_err = translationError(pose_error);

	_err.push_back(errors(r_err, t_err));
}

void GetStats(std::vector<errors> err, double& t_error, double& r_error) {
	float t_err = 0;
	float r_err = 0;

	// for all errors do => compute sum of t_err, r_err
	for (std::vector<errors>::iterator it = err.begin(); it != err.end(); it++) {
		t_err += it->t_err;
		r_err += it->r_err;
	}

	float num = err.size();

	t_error = /*100.0 **/ t_err / num;
	r_error = /*100.0 **/ r_err / num;
}

inline static void
initializePose(const int &frameID, struct VisualOdometry &vo)
{
	getAbsoluteTransformationMat(frameID, vo.T);
	vo.T_messure = vo.T.clone();
	//vo.poses_gt.push_back(vo.T.clone());
	vo.poses_messure.push_back(vo.T.clone());
	//T2Rt(vo.T, vo.R, vo.t);

	Mat P = makeProjectionMatrix(vo.T);

	vo.P_Vec.push_back(P.clone());
	vo.P_Keyframe = P.clone();
}

bool
addFrame(struct Frame &frame)
{
	if (!getFrame(frame, frameNum)) {
		printf("Added frame: %d error !!! \n \n", frameNum);
		return false;
	}
	frameNum += frameCounter;

	return true;
}

static void 
showTrackingTimeStatistics()
{
	// Tracking time statistics
	sort(vTimesTrack.begin(), vTimesTrack.end());
	float totaltime = 0;
	for (int ni = 0; ni < vTimesTrack.size(); ni++)
	{
		totaltime += vTimesTrack[ni];
	}
	cout << "-------" << endl << endl;
	cout << "median tracking time: " << vTimesTrack[vTimesTrack.size() / 2] << endl;
	cout << "mean tracking time: " << totaltime / vTimesTrack.size() << endl;
	cout << "min tracking time: " << vTimesTrack[0] << endl;
	cout << "max tracking time: " << vTimesTrack[vTimesTrack.size() - 1] << endl;
}

static struct Frame
odometryInitial(struct VisualOdometry &vo, struct Frame &lastFrame)
{
	if (!addFrame(lastFrame))
		return lastFrame;
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);
	vo.ptSizeKF_TH = lastFrame.pt.size();
	vo.ptKF = lastFrame.pt;

	bool correctPose = false;
	while (!correctPose) {
		struct Frame currentFrame;
		if (!addFrame(currentFrame)) {
			waitKey(0);
		}
		bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg,
										 lastFrame.pt, currentFrame.pt,
										 vo.ptKF, vo.ptSizeKF_TH, vo.points3D_Vec, false);

		Mat T_f;
		bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.ptKF, vo.points3D_Vec, T_f, false);

		if (isCorrect) {
			cout << "Frame: " << currentFrame.id << endl;

			double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[lastFrame.id];
			updataPose(scale, T_f, vo.T_messure);

			return currentFrame;
		}
	}
}

static struct Frame
odometryInitial(struct VisualOdometry &vo, struct Frame &lastFrame, vector<uchar> &status)
{
	if (!addFrame(lastFrame))
		return lastFrame;
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);
	vo.ptSizeKF_TH = lastFrame.pt.size();
	vo.ptKF = lastFrame.pt;

	bool correctPose = false;
	while (!correctPose) {
		struct Frame currentFrame;
		if (!addFrame(currentFrame)) {
			waitKey(0);
		}
		const int initialKF_TH = 200;
		bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg,
										 lastFrame.pt, currentFrame.pt, status, vo.ptSizeKF_TH);

		vo.status_after_tracking = vo.status;

		Mat T_f;
		bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, status, T_f, true);

		if (isCorrect) {
			cout << "Initial with Frame: " << currentFrame.id << endl;

			double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[lastFrame.id];
			updataPose(scale, T_f, vo.T_messure);

			return currentFrame;
		}
	}
}

/*void 
odometryRun() 
{
	frameNum = firstFrameID;
	int keyFrameSize = 0;
	bool findKeyFrame = false;

	struct VisualOdometry vo;
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);

	struct Frame lastFrame;
	if (!addFrame(lastFrame)) {
		return;
	}
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);
	keyFrameSize = lastFrame.pt.size();

	initializePose(lastFrame.id, vo);

	while (true) {
		loop_start = clock();

		struct Frame currentFrame;
		if (!addFrame(currentFrame)) {
			waitKey(1);
			break;
		}

		if (//lastFrame.pt.size() < MIN_NUM_FEAT && 
		findKeyFrame) {
			findKeyFrame = false;
			//cout << "Redetect!" << endl;
			detectKeypoints(lastFrame.grayImg, lastFrame.pt);
			keyFrameSize = lastFrame.pt.size();
		}

		bool keyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg, lastFrame.pt, currentFrame.pt, keyFrameSize);

		//cout << "last: " << lastFrame.id << "   current: " << currentFrame.id << endl;
		//bool correct = poseEstimate(lastFrame.pt, currentFrame.pt, currentFrame.R_f, currentFrame.t_f, keyFrame);
		bool correct = poseEstimate(lastFrame.pt, currentFrame.pt, currentFrame.T, keyFrame);
		
		//getline(poseFile, textLine);
		if (correct) {
			findKeyFrame = true;

			getAbsoluteTransformationMat(currentFrame.id, vo.T);
			scale = getAbsoluteScale(vo.T);
			//scale = calcAbsoluteScale();
			//cout << scale << endl << endl;

			updataPose(scale, vo, currentFrame.T);
			//vo.t = vo.t + scale * (vo.R * currentFrame.t_f);
			//vo.R = vo.R * currentFrame.R_f;
		}
		loop_end = clock();

		showFrame(lastFrame, currentFrame);
		showTrajAndGTruth(traj, vo.T_messure, vo.T);
		//showTraj(traj, vo.t);

		if (correct) {
			lastFrame = currentFrame;
		}

		waitKey(1);
	}
}*/

void
odometryRun()
{
	frameNum = firstFrameID;
	int keyFramePtSizeTH = 0;
	bool findKeyFrame = false;

	struct VisualOdometry vo;
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);

	struct Frame lastFrame;
	if (!addFrame(lastFrame)) {
		return;
	}
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);
	keyFramePtSizeTH = lastFrame.pt.size();
	vo.ptKF = lastFrame.pt;

	initializePose(lastFrame.id, vo);

	while (true) {
		loop_start = clock();

		struct Frame currentFrame;
		if (!addFrame(currentFrame)) {
			waitKey(1);
			break;
		}

		if (//lastFrame.pt.size() < MIN_NUM_FEAT &&
			findKeyFrame) {
			findKeyFrame = false;
			//cout << "Redetect!" << endl;
			detectKeypoints(lastFrame.grayImg, lastFrame.pt);
			keyFramePtSizeTH = lastFrame.pt.size();
			vo.ptKF = lastFrame.pt;
		}

		bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg, lastFrame.pt, currentFrame.pt, vo.ptKF, keyFramePtSizeTH);

		//cout << "last: " << lastFrame.id << "   current: " << currentFrame.id << endl;
		//bool correct = poseEstimate(lastFrame.pt, currentFrame.pt, currentFrame.R_f, currentFrame.t_f, keyFrame);
		bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.ptKF, currentFrame.T, isKeyFrame);

		//getline(poseFile, textLine);
		if (isCorrect) {
			findKeyFrame = true;

			getAbsoluteTransformationMat(currentFrame.id, vo.T);
			scale = getAbsoluteScale(vo.T);
			//scale = calcAbsoluteScale();
			//cout << scale << endl << endl;

			updataPose(scale, vo, currentFrame.T);
			//vo.t = vo.t + scale * (vo.R * currentFrame.t_f);
			//vo.R = vo.R * currentFrame.R_f;
		}
		loop_end = clock();

		showFrame(lastFrame, currentFrame);
		showTrajAndGTruth(traj, vo.T_messure, vo.T);
		//showTraj(traj, vo.t);


		lastFrame = currentFrame;

		waitKey(1);
	}
}

/*void
odometryRunLBP()
{
	frameNum = firstFrameID;
	int keyFramePtSizeTH = 0, keyFramePtSizeTH2 = 0;
	bool findKeyFrame = false;
	//bool findKeyFrame = true;

	struct VisualOdometry vo;
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);
	int use = 0;
	double R_err = 0, t_err = 0, r_errCurr = 0, t_errCurr = 0;

	struct Frame lastFrame;
	if (!addFrame(lastFrame)) {
		return;
	}
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);
	keyFramePtSizeTH = lastFrame.pt.size();
	vo.ptKF = lastFrame.pt;

	initializePose(lastFrame.id, vo);

//	add LBP for test
	struct VisualOdometry vo_LBP;
	loadPoses(vo_LBP.poses_gt);
	trajectoryDistances(vo_LBP.poses_gt, vo_LBP.trajDistances);
	initializePose(lastFrame.id, vo_LBP);

	struct LBP_Feature lastFrameLBP;
	gen_LBP_Frame(lastFrame.grayImg, lastFrameLBP.lbpFrame);
	//detectKeypointsLBP(lastFrameLBP.lbpFrame, lastFrameLBP.lbpPt);
	//goodFeaturesToTrack(lastFrameLBP.lbpFrame, lastFrameLBP.lbpPt, 3000, 0.05, 15, Mat(), 5, true, 0.04);	// 3000, 0.05, 15, 5, 0.04
	goodFeaturesToTrack(lastFrameLBP.lbpFrame, lastFrameLBP.lbpPt, 3000, 0.01, 20, Mat(), 5, true, 0.06);		// 3000, 0.01, 20, 5, 0.06
	//vector<Point2f> pt;
	//createGrid(pt, lastFrame.grayImg.cols, lastFrame.grayImg.rows, 50, 10);
	//createGrid(pt, lastFrame.grayImg.rows, lastFrame.grayImg.cols, 10, 5);
	//createGrid(pt, lastFrame.grayImg.rows, lastFrame.grayImg.rows, 10, 5);
	//lastFrameLBP.lbpPt = pt;
	//lastFrameLBP.lbpPt = lastFrame.pt;
	keyFramePtSizeTH2 = lastFrameLBP.lbpPt.size();
	vo_LBP.ptKF = lastFrameLBP.lbpPt;
	int kfID = lastFrame.id;
	//gen_lbpMat(lastFrameLBP.lbpFrame, lastFrameLBP.lbpPt, lastFrameLBP.lbpMat);
//	end-------------

	while (true) {
		loop_start = clock();

		struct Frame currentFrame;
		if (!addFrame(currentFrame)) {
			waitKey(1);
			break;
		}

//	add LBP for test
		struct LBP_Feature currentFrameLBP;
		gen_LBP_Frame(currentFrame.grayImg, currentFrameLBP.lbpFrame);
//	end-------------

		if (//lastFrameLBP.lbpPt.size() < MIN_NUM_FEAT ||
		findKeyFrame) {
			findKeyFrame = false;
			//cout << "Redetect!" << endl;
			detectKeypoints(lastFrame.grayImg, lastFrame.pt);
			keyFramePtSizeTH = lastFrame.pt.size();
			vo.ptKF = lastFrame.pt;

//	add LBP for test
			//detectKeypointsLBP(lastFrameLBP.lbpFrame, lastFrameLBP.lbpPt);
			//goodFeaturesToTrack(lastFrameLBP.lbpFrame, lastFrameLBP.lbpPt, 3000, 0.05, 15, Mat(), 5, true, 0.04);
			goodFeaturesToTrack(lastFrameLBP.lbpFrame, lastFrameLBP.lbpPt, 3000, 0.01, 20, Mat(), 5, true, 0.06);
			//lastFrameLBP.lbpPt = pt;
			//lastFrameLBP.lbpPt = lastFrame.pt;
			keyFramePtSizeTH2 = lastFrameLBP.lbpPt.size();
			vo_LBP.ptKF = lastFrameLBP.lbpPt;
			kfID = lastFrame.id;
			//gen_lbpMat(lastFrameLBP.lbpFrame, lastFrameLBP.lbpPt, lastFrameLBP.lbpMat);
//	end-------------
		}

		bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg, lastFrame.pt, currentFrame.pt, vo.ptKF, keyFramePtSizeTH);

//	add LBP for test
		//trackLBPMat(currentFrameLBP.lbpFrame, lastFrameLBP.lbpPt, lastFrameLBP.lbpMat, currentFrameLBP.lbpPt);
		bool isKeyFrame2 = trackKeypoints(lastFrameLBP.lbpFrame, currentFrameLBP.lbpFrame, lastFrameLBP.lbpPt, currentFrameLBP.lbpPt, vo_LBP.ptKF, keyFramePtSizeTH2);
		/*if (findKeyFrame) { 
			findKeyFrame = false;
			keyFramePtSizeTH = lastFrame.pt.size();
			vo.ptKF = lastFrame.pt;

			keyFramePtSizeTH2 = lastFrameLBP.lbpPt.size();
			vo_LBP.ptKF = lastFrameLBP.lbpPt;
			kfID = lastFrame.id;
		}
//	end-------------

		//cout << "last: " << lastFrame.id << "   current: " << currentFrame.id << endl;
		//bool correct = poseEstimate(lastFrame.pt, currentFrame.pt, currentFrame.R_f, currentFrame.t_f, keyFrame);
		bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.ptKF, currentFrame.T, isKeyFrame);

//	add LBP for test
		Mat T;
		bool isCorrect2 = poseEstimate(lastFrameLBP.lbpPt, currentFrameLBP.lbpPt, vo_LBP.ptKF, T, isKeyFrame2);
//	end-------------


		//getline(poseFile, textLine);
		if (isCorrect2) {
			findKeyFrame = true;

			//getAbsoluteTransformationMat(currentFrame.id, vo_LBP.T);
			vo_LBP.T = vo_LBP.poses_gt[currentFrame.id];
			//vo_LBP.poses_gt.push_back(vo_LBP.T.clone());
			//scale = getAbsoluteScale(vo_LBP.T);
			scale = vo_LBP.trajDistances[currentFrame.id] - vo_LBP.trajDistances[kfID];
			//scale = calcAbsoluteScale();
			//cout << scale << endl << endl;

			//updataPose(scale, vo, currentFrame.T);
			//vo.t = vo.t + scale * (vo.R * currentFrame.t_f);
			//vo.R = vo.R * currentFrame.R_f;

//	add LBP for test
			updataPose(scale, vo_LBP, T);
			vo_LBP.poses_messure.push_back(vo_LBP.T_messure.clone());
//	end-------------
			CalcSequenceErrors(vo_LBP.poses_gt, vo_LBP.poses_messure, vo_LBP.err, kfID, currentFrame.id, scale);
			use = vo_LBP.err.size() - 1;
			r_errCurr = vo_LBP.err[use].r_err;
			t_errCurr = vo_LBP.err[use].t_err;
			//Mat t_gt = vo_LBP.T(Range(0, 3), Range(3, 4));
			//Mat t_messure = vo_LBP.T_messure(Range(0, 3), Range(3, 4));
			//t_errCurr = norm(t_gt, t_messure, NORM_L2);
			GetStats(vo_LBP.err, t_err, R_err);

			// make projection matrix
			/*Mat R = vo_LBP.T_messure(Range(0, 3), Range(0, 3));
			Mat t = vo_LBP.T_messure(Range(0, 3), Range(3, 4));

			Mat P(3, 4, CV_64F);

			P(Range(0, 3), Range(0, 3)) = R.t();
			P(Range(0, 3), Range(3, 4)) = -R.t()*t;
			P = K * P;
			vo_LBP.P.push_back(P.clone());

			Mat points4D;
			triangulatePoints(vo_LBP.P[use], vo_LBP.P[use + 1], lastFrameLBP.lbpPt, currentFrameLBP.lbpPt, points4D);
		}
		else {
			//cout << "!!!! \n\n" << endl;
		}
		loop_end = clock();

		//showFrame(lastFrame, currentFrame);

		//Mat show3;
		//vector<KeyPoint> keypoints3;
		//KeyPoint::convert(lastFrame.pt, keypoints3);
		//drawKeypoints(currentFrame.img, keypoints3, show3, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show3, lastFrame.pt, currentFrame.pt);
	
		Mat show2;
		vector<KeyPoint> keypoints;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(lastFrameLBP.lbpPt, keypoints);
		drawKeypoints(currentFrame.img, keypoints, show2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(show2, lastFrameLBP.lbpPt, currentFrameLBP.lbpPt);

		putText(show2, "FPS: " + to_string((1 / ((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)))) +
					   "  R_err: " + to_string(r_errCurr) + 
					   "  t_err: " + to_string(t_errCurr) + 
					   "  Error_R: " + to_string(R_err) + 
					   "  Error_t: " + to_string(t_err), Point(5, lastFrame.img.rows - 40), 1, 1, Scalar(0, 255, 0), 2);
		imshow("test", show2);

		//showTrajAndGTruth(traj, vo.T_messure, vo.T);
		showTrajAndGTruth(traj, vo_LBP.T_messure, vo_LBP.T);
		//showTraj(traj, vo.t);

		lastFrame = currentFrame;

		lastFrameLBP = currentFrameLBP;


		waitKey(1);
	}
}*/

void 
odometryRun_PNP() 
{
	//--- set first frame ID number...
	frameNum = firstFrameID;

	//--- prepare VO for using...
	struct VisualOdometry vo;
	loadPoses(vo.poses_gt);
	trajectoryDistances(vo.poses_gt, vo.trajDistances);
	initializePose(firstFrameID, vo);
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);
	bool findKeyframe = false;
	int keyframeCounter = 0;
	int use = 0;
	double R_err = 0, t_err = 0, r_errCurr = 0, t_errCurr = 0;
	
	double minFPS = 999, maxFPS =0, avgFPS = 0;
	vector<double> FPS_vec;

	//--- prepare for write pose file...
	//string pose_gt_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_gt_Runtime.txt";
	//string pose_em_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_em_Runtime.txt";
	//string pose_gt_Runtime = "..\\pose_gt_Runtime.txt";
	//string pose_em_Runtime = "..\\pose_em_Runtime.txt";

	string gt_dir = "..\\poses";
	string result__data_dir = "..\\poses\\results\\data";
	// create output directories
	system(("mkdir " + gt_dir).c_str());
	system(("mkdir " + result__data_dir).c_str());

	string pose_gt_Runtime = gt_dir + "\\00.txt";
	string pose_em_Runtime = result__data_dir + "\\00.txt";

	ofstream out_gt(pose_gt_Runtime);
	ofstream out_em(pose_em_Runtime);

	for (int i = 0; i < vo.T.rows - 1; i++) {
		for (int j = 0; j< vo.T.cols; j++) {
			out_gt << vo.T.at<double>(i, j) << " ";
		}
	}
	out_gt << endl;

	for (int i = 0; i < vo.T_messure.rows - 1; i++) {
		for (int j = 0; j< vo.T_messure.cols; j++) {
			out_em << vo.T_messure.at<double>(i, j) << " ";
		}
	}
	out_em << endl;


#ifdef write_Video
	writer.open(filename, codec, fps, videoSize);
#endif // write_Video


	//--- prepare first frame...
	struct Frame lastFrame;
	if (!addFrame(lastFrame))
		return;
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);


	//--- feature detector ---
	//goodFeaturesToTrack(lastFrame.grayImg, lastFrame.pt, 264, qualityLevel, minDistance, Mat(), detectBlockSize, false, 0.04);
	//Ptr<ORB> detector = ORB::create(5000);
	//Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
	//Ptr<Feature2D> detector = xfeatures2d::SURF::create();

	//detector->detect(lastFrame.grayImg, lastFrame.keypoints);
	//KeyPoint::convert(lastFrame.keypoints, lastFrame.pt);
	//------------------


	vo.ptSizeKF_TH = lastFrame.pt.size();
	vo.ptKF = lastFrame.pt;
	int kfID = lastFrame.id;

	bool have_3DPt = false;
	while (true) {
		//--- add next frame and prepare for using...
		struct Frame currentFrame;
		if (!addFrame(currentFrame)) {
			out_gt.close();
			out_em.close();
			showTrackingTimeStatistics();
			//waitKey(0);
			break;
		}
#ifdef use_flag
		if (currentFrame.id == 152)	continue;	// KITTI_09 seems bad frame.
#endif // use_flag

		loop_start = clock();

		if (findKeyframe) {
			findKeyframe = false;

			detectKeypoints(lastFrame.grayImg, lastFrame.pt);

			//--- feature detector ---
			//goodFeaturesToTrack(lastFrame.grayImg, lastFrame.pt, 264, qualityLevel, minDistance, Mat(), detectBlockSize, false, 0.04);
			//detector->detect(lastFrame.grayImg, lastFrame.keypoints);
			//KeyPoint::convert(lastFrame.keypoints, lastFrame.pt);
			//------------------

			vo.ptSizeKF_TH = lastFrame.pt.size();
			vo.ptKF = lastFrame.pt;
			kfID = lastFrame.id;
		}

		//--- track kp in lbp frame and check if find key frame...
		bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg,
										 lastFrame.pt, currentFrame.pt,
										 vo.ptKF, vo.ptSizeKF_TH, vo.points3D_Vec, have_3DPt);

		/*Mat showTest;
		vector<KeyPoint> keypointsTest;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.lbp_feature.lbpPt, keypointsTest);
		drawKeypoints(currentFrame.img, keypointsTest, showTest, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(showTest, lastFrame.lbp_feature.lbpPt, currentFrame.lbp_feature.lbpPt);
		imshow("test", showTest);*/
		//waitKey(0);

		//--- calculate rotation and translation, check if pose correct...
		Mat T_f;
		bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.ptKF, vo.points3D_Vec, T_f, have_3DPt);
		//bool isCorrect = poseEstimate(voLBP.ptKF, currentFrame.lbp_feature.lbpPt, lastFrame.lbp_feature.lbpPt, voLBP.points3D_Vec, T_f, have_3DPt);

		/*Mat showTest2;
		vector<KeyPoint> keypointsTest2;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.lbp_feature.lbpPt, keypointsTest2);
		drawKeypoints(currentFrame.img, keypointsTest2, showTest2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(showTest2, lastFrame.lbp_feature.lbpPt, currentFrame.lbp_feature.lbpPt);
		imshow("test", showTest2);*/

		/*vector<errors> err_f;
		double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[kfID];
		Mat T_f_temp = T_f.clone();
		Mat t = T_f_temp(Range(0, 3), Range(3, 4));
		t *= scale;
		poseFrameErrors(vo.poses_gt, T_f_temp, kfID, currentFrame.id, err_f);
		printf("R: %f   t: %f \n", err_f[0].r_err, err_f[0].t_err);*/


		if (isCorrect) {
			have_3DPt = true;

			double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[kfID];
			Mat T = vo.T_messure.clone();
			updataPose(scale, T_f, T);

			//--- make projection matrix
			Mat P = makeProjectionMatrix(T);
			//voLBP.P_Vec.push_back(P.clone());

			Mat points4D;
			//triangulatePoints(voLBP.P_Vec[keyframeCounter], P, voLBP.ptKF, currentFrame.lbp_feature.lbpPt, points4D);
			triangulatePoints(vo.P_Keyframe, P, vo.ptKF, currentFrame.pt, points4D);
			//triangulatePoints(voLBP.P_Vec[currentFrame.id], voLBP.P_Vec[kfID], currentFrame.lbp_feature.lbpPt, voLBP.ptKF, points4D);
			
			//--- normalize 4D points to 3D points and save to pt3DVec...
			vector<Point3d> points3D;
			normalizePoints(points4D, points3D);
			vo.points3D_Vec.push_back(points3D);

			//--- testing
			// Chierality check
			vector<uchar> status;
			for (unsigned int i = 0; i < points3D.size(); i++)
				status.push_back((points3D[i].z > 0) ? 1 : 0);

			//cout << "triangulation likes " << countNonZero(Mat(status))
			//	 << " out of " << currentFrame.pt.size()
			//	 << " (" << (float)(countNonZero(Mat(status))) / (float)(currentFrame.pt.size()) * 100 << "%)" << endl << endl;

			//showTriangulation(voLBP.ptKF, currentFrame.lbp_feature.lbpPt, R, t, points3D);

			if (isKeyFrame) {
				//printf("keyframe: %d \n", currentFrame.id);
				findKeyframe = true;
				have_3DPt = false;
				keyframeCounter++;

				vector<Point3d> pt3D_Result;
				vector<Point2f> pt2D_Result;
				const int ptSize = vo.ptKF.size();
				const int vecSize = vo.points3D_Vec.size();
				for (int i = 0; i < ptSize; i++) {
					/*Point3d p(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
						p += voLBP.points3D_Vec[j].at(i);
					}
					p /= vecSize;

					bool push = false;
					int counter = 0;
					Point3d p_new(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
						double deltaX = abs( (p - voLBP.points3D_Vec[j].at(i)).x );
						double deltaY = abs( (p - voLBP.points3D_Vec[j].at(i)).y );

						if (deltaX <= 0.1 && deltaY <= 0.1) {
							push = true;
							p_new += voLBP.points3D_Vec[j].at(i);
							counter++;
						}
					}
					if (push) {
						p_new /= counter;
						pt3D_Result.push_back(p_new);
						pt2D_Result.push_back(currentFrame.lbp_feature.lbpPt.at(i));
					}*/

					Point3d mu(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
						mu += vo.points3D_Vec[j].at(i);
					}
					mu /= vecSize;

					Point3d Sigma(0, 0, 0);
					vector<Point3d> deltaVec;
					for (int j = 0; j < vecSize; j++) {
						const double deltaX = abs((mu - vo.points3D_Vec[j].at(i)).x);
						const double deltaY = abs((mu - vo.points3D_Vec[j].at(i)).y);
						const double deltaZ = abs((mu - vo.points3D_Vec[j].at(i)).z);
						deltaVec.push_back(Point3d(deltaX, deltaY, deltaZ));

						Sigma.x += pow(deltaX, 2);
						Sigma.y += pow(deltaY, 2);
						Sigma.z += pow(deltaZ, 2);
					}
					Sigma /= vecSize;
					Sigma = Point3d(sqrtf(Sigma.x), sqrtf(Sigma.y), sqrtf(Sigma.z));

					bool push = false;
					int counter = 0;
					Point3d p_new(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
						Point3d delta = deltaVec[j];
						const double errScoreX = delta.x / Sigma.x;
						const double errScoreY = delta.y / Sigma.y;
						const double errScoreZ = delta.z / Sigma.z;

						if (errScoreX < 1.2 && errScoreY < 1.2 && errScoreZ < 1.2) {
							push = true;
							p_new += vo.points3D_Vec[j].at(i);
							counter++;
						}
					}

					if (push) {
						p_new /= counter;
						pt3D_Result.push_back(p_new);
						pt2D_Result.push_back(currentFrame.pt.at(i));
					}
					else {
						//printf("error too large, use mu !!! \n\n");
						pt3D_Result.push_back(mu);
						pt2D_Result.push_back(currentFrame.pt.at(i));
					}
				}
				//voLBP.points3D_Vec.clear();
				if (pt3D_Result.size() <= 4) {
					cout << "no pass !!!" << endl;
				}

				Mat r_pnp, t_pnp, R_pnp, mask_PNP;
				//solvePnP(vo.points3D_Vec[vecSize-1], currentFrame.pt, K, Mat(), r_pnp, t_pnp, false, SOLVEPNP_ITERATIVE);
				//solvePnPRansac(pt3D_Result, currentFrame.lbp_feature.lbpPt, K, Mat(), r_pnp, t_pnp, false, 1000, 1, 0.99);
				solvePnPRansac(pt3D_Result, pt2D_Result, K, Mat(), r_pnp, t_pnp, false, iterationsCount, reprojectionError, 0.995);
				//solvePnPRansac(pt3D_Result, pt2D_Result, K, Mat(), r_pnp, t_pnp, false, 5000, 1.0, 0.995, mask_PNP, SOLVEPNP_AP3P);
				//solvePnPRansac(vo.points3D_Vec[vecSize-1], currentFrame.pt, K, Mat(), r_pnp, t_pnp, false, 1000, 1.0, 0.99);
				
				//--- inverse transformation from solvePnP...
				Mat T_pnp = inverseTransformation(r_pnp, t_pnp);

				//--- make projection matrix using transformation matrix...
				Mat P_final = makeProjectionMatrix(T_pnp);
				//voLBP.P_Vec.push_back(P_final.clone());
				vo.P_Keyframe = P_final.clone();

				//updataPose(scale, voLBP, T_pnp);
				//voLBP.poses_messure.push_back(voLBP.T_messure.clone());
				vo.T = vo.poses_gt[currentFrame.id];
				vo.T_messure = T_pnp.clone();
				vo.poses_messure.push_back(T_pnp.clone());

				CalcSequenceErrors(vo.poses_gt, vo.poses_messure, vo.err, kfID, currentFrame.id, scale);
				use = vo.err.size() - 1;
				r_errCurr = vo.err[use].r_err / scale;
				t_errCurr = vo.err[use].t_err / scale;
				GetStats(vo.err, t_err, R_err);

				for (int i = 0; i < vo.T.rows - 1; i++) {
					for (int j = 0; j< vo.T.cols; j++) {
						out_gt << vo.T.at<double>(i, j) << " ";
					}
				}
				out_gt << endl;

				for (int i = 0; i < vo.T_messure.rows - 1; i++) {
					for (int j = 0; j< vo.T_messure.cols; j++) {
						out_em << vo.T_messure.at<double>(i, j) << " ";
					}
				}
				out_em << endl;

				if (r_errCurr >= 0.005) {
					//printf("Rotation error is large !!! \n");
				}
				//cout << vo.points3D_Vec.size() << endl << endl;
				vo.points3D_Vec.clear();
				//printf("kfID: %d    frameID: %d \n", kfID, currentFrame.id);
			}
		}
		else {
			//cout << "next frame !!! \n" << endl;
		}
		//bundleAdjustment(points3D, currentFrame.lbp_feature.lbpPt, K, R, t);
		loop_end = clock();

		vTimesTrack.push_back( (loop_end - loop_start) / (double)(CLOCKS_PER_SEC) );

		double fps = 1 / ( (loop_end - loop_start) / (double)(CLOCKS_PER_SEC) );
		FPS_vec.push_back(fps);
		if (fps < minFPS) minFPS = fps;
		if (fps > maxFPS) maxFPS = fps;
		double sum = std::accumulate(FPS_vec.begin(), FPS_vec.end(), 0.0);
		avgFPS = sum / FPS_vec.size();


		Mat show2;
		vector<KeyPoint> keypoints;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.pt, keypoints);
		drawKeypoints(currentFrame.img, keypoints, show2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(show2, lastFrame.pt, currentFrame.pt);

		putText(show2, //"FPS: " + to_string((1 / ((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)))) +
			"  ID: " + to_string(currentFrame.id) +
			//"  R_err: " + to_string(r_errCurr) +
			//"  t_err: " + to_string(t_errCurr) +
			//"  Error_R: " + to_string(R_err) +
			//"  Error_t: " + to_string(t_err) + 
			"  Time: " + to_string((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)) + 
			"(FPS: " + to_string((1 / ((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)))) + ")" + 
			"  minFPS: " + to_string(minFPS) +
			"  maxFPS: " + to_string(maxFPS) + 
			"  avgFPS: " + to_string(avgFPS), Point(5, lastFrame.img.rows - 40), 1, 1, Scalar(0, 255, 0), 2);
		imshow("test", show2);

		showTrajAndGTruth(traj, vo.T_messure, vo.T);


#ifdef write_Video
		writer.write(show2);
#endif // write_Video


		lastFrame = currentFrame;
		
		waitKey(1);
	}
}

void
odometryRun_ALL_PNP()
{
	//--- set first frame ID number...
	frameNum = firstFrameID;

	//--- prepare VO for using...
	struct VisualOdometry vo;
	loadPoses(vo.poses_gt);
	trajectoryDistances(vo.poses_gt, vo.trajDistances);
	initializePose(firstFrameID, vo);
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);
	bool findKeyframe = false;
	int keyframeCounter = 0;
	int use = 0;
	double R_err = 0, t_err = 0, r_errCurr = 0, t_errCurr = 0;

	//--- prepare for write pose file...
	string pose_gt_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_gt_Runtime.txt";
	string pose_em_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_em_Runtime.txt";
	ofstream out_gt(pose_gt_Runtime);
	ofstream out_em(pose_em_Runtime);

	for (int i = 0; i < vo.T.rows - 1; i++) {
		for (int j = 0; j< vo.T.cols; j++) {
			out_gt << vo.T.at<double>(i, j) << " ";
		}
	}
	out_gt << endl;

	for (int i = 0; i < vo.T_messure.rows - 1; i++) {
		for (int j = 0; j< vo.T_messure.cols; j++) {
			out_em << vo.T_messure.at<double>(i, j) << " ";
		}
	}
	out_em << endl;

	//--- run odometry initial...
	struct Frame lastFrame;
	struct Frame currentFrame = odometryInitial(vo, lastFrame);

	vo.poses_messure.push_back(vo.T_messure.clone());
	int kfID = lastFrame.id;

	//--- make projection matrix
	currentFrame.P = makeProjectionMatrix(vo.T_messure);

	Mat points4D;
	triangulatePoints(vo.P_Keyframe, currentFrame.P, vo.ptKF, currentFrame.pt, points4D);
	bool have_3DPt = true;

	//--- normalize 4D points to 3D points and save to pt3DVec...
	vector<Point3d> points3D;
	normalizePoints(points4D, points3D);
	vo.points3D_Vec.push_back(points3D);

	//--- Chierality check
	vector<uchar> status;
	for (unsigned int i = 0; i < points3D.size(); i++)
		status.push_back((points3D[i].z > 0) ? 1 : 0);

	cout << "triangulation likes " << countNonZero(Mat(status))
		 << " out of " << currentFrame.pt.size()
		 << " (" << (float)(countNonZero(Mat(status))) / (float)(currentFrame.pt.size()) * 100 << "%)" << endl << endl;

	// remove bad points.
	remove_bad_points(lastFrame.pt, status);
	remove_bad_points(currentFrame.pt, status);
	remove_bad_points(vo.ptKF, status);
	remove_bad_points(vo.points3D_Vec, status);

	//lastFrame = currentFrame;

	while (true) {
		//--- add next frame and prepare for using...
		struct Frame nextFrame;
		if (!addFrame(nextFrame)) {
			out_gt.close();
			out_em.close();
			waitKey(0);
			break;
		}

		if (findKeyframe) {
			findKeyframe = false;

			detectKeypoints(lastFrame.grayImg, lastFrame.pt);

			vo.ptSizeKF_TH = lastFrame.pt.size();
			vo.ptKF = lastFrame.pt;
			kfID = lastFrame.id;

			bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg,
											 lastFrame.pt, currentFrame.pt,
											 vo.ptKF, vo.ptSizeKF_TH, vo.points3D_Vec, false);

			Mat T_f;
			bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.ptKF, vo.points3D_Vec, T_f, false);

			Mat points4D;
			triangulatePoints(lastFrame.P, currentFrame.P, vo.ptKF, currentFrame.pt, points4D);

			//--- normalize 4D points to 3D points and save to pt3DVec...
			vector<Point3d> points3D;
			normalizePoints(points4D, points3D);
			vo.points3D_Vec.push_back(points3D);

			//--- Chierality check
			vector<uchar> status;
			for (unsigned int i = 0; i < points3D.size(); i++)
				status.push_back((points3D[i].z > 0) ? 1 : 0);

			cout << "triangulation likes " << countNonZero(Mat(status))
				<< " out of " << currentFrame.pt.size()
				<< " (" << (float)(countNonZero(Mat(status))) / (float)(currentFrame.pt.size()) * 100 << "%)" << endl << endl;

			// remove bad points.
			remove_bad_points(lastFrame.pt, status);
			remove_bad_points(currentFrame.pt, status);
			remove_bad_points(vo.ptKF, status);
			remove_bad_points(vo.points3D_Vec, status);
		}

		//--- track kp in lbp frame and check if find key frame...
		bool isKeyFrame = trackKeypoints(currentFrame.grayImg, nextFrame.grayImg,
										 currentFrame.pt, nextFrame.pt,
										 vo.ptKF, vo.ptSizeKF_TH, vo.points3D_Vec, have_3DPt);

		/*Mat showTest;
		vector<KeyPoint> keypointsTest;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.lbp_feature.lbpPt, keypointsTest);
		drawKeypoints(currentFrame.img, keypointsTest, showTest, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(showTest, lastFrame.lbp_feature.lbpPt, currentFrame.lbp_feature.lbpPt);
		imshow("test", showTest);*/
		//waitKey(0);

		//--- calculate rotation and translation, check if pose correct...
		Mat T_f;
		bool isCorrect = poseEstimate(currentFrame.pt, nextFrame.pt, vo.ptKF, vo.points3D_Vec, T_f, have_3DPt);

		/*Mat showTest2;
		vector<KeyPoint> keypointsTest2;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.lbp_feature.lbpPt, keypointsTest2);
		drawKeypoints(currentFrame.img, keypointsTest2, showTest2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(showTest2, lastFrame.lbp_feature.lbpPt, currentFrame.lbp_feature.lbpPt);
		imshow("test", showTest2);*/

		/*vector<errors> err_f;
		double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[kfID];
		Mat T_f_temp = T_f.clone();
		Mat t = T_f_temp(Range(0, 3), Range(3, 4));
		t *= scale;
		poseFrameErrors(vo.poses_gt, T_f_temp, kfID, currentFrame.id, err_f);
		printf("R: %f   t: %f \n", err_f[0].r_err, err_f[0].t_err);*/

		Mat r_pnp, t_pnp, R_pnp;
		//solvePnP(vo.points3D_Vec[vo.points3D_Vec.size()-1], nextFrame.pt, K, Mat(), r_pnp, t_pnp, false, SOLVEPNP_ITERATIVE);
		//solvePnPRansac(pt3D_Result, currentFrame.lbp_feature.lbpPt, K, Mat(), r_pnp, t_pnp, false, 1000, 1, 0.99);
		//solvePnPRansac(pt3D_Result, pt2D_Result, K, Mat(), r_pnp, t_pnp, false, 5000, 1.0, 0.995);
		solvePnPRansac(vo.points3D_Vec[vo.points3D_Vec.size() - 1], nextFrame.pt, K, Mat(), r_pnp, t_pnp, false, 10000, 3.0, 0.99);

		//--- inverse transformation from solvePnP...
		Mat T_pnp = inverseTransformation(r_pnp, t_pnp);

		//--- make projection matrix using transformation matrix...
		Mat P_final = makeProjectionMatrix(T_pnp);
		nextFrame.P = P_final.clone();
		//voLBP.P_Vec.push_back(P_final.clone());
		vo.P_Keyframe = P_final.clone();

		//updataPose(scale, voLBP, T_pnp);
		//voLBP.poses_messure.push_back(voLBP.T_messure.clone());
		vo.T = vo.poses_gt[nextFrame.id];
		vo.T_messure = T_pnp.clone();
		vo.poses_messure.push_back(T_pnp.clone());

		CalcSequenceErrors(vo.poses_gt, vo.poses_messure, vo.err, kfID, nextFrame.id, scale);
		use = vo.err.size() - 1;
		r_errCurr = vo.err[use].r_err / scale;
		t_errCurr = vo.err[use].t_err / scale;
		GetStats(vo.err, t_err, R_err);

		for (int i = 0; i < vo.T.rows - 1; i++) {
			for (int j = 0; j< vo.T.cols; j++) {
				out_gt << vo.T.at<double>(i, j) << " ";
			}
		}
		out_gt << endl;

		for (int i = 0; i < vo.T_messure.rows - 1; i++) {
			for (int j = 0; j< vo.T_messure.cols; j++) {
				out_em << vo.T_messure.at<double>(i, j) << " ";
			}
		}
		out_em << endl;

		if (r_errCurr >= 0.005) {
			//printf("Rotation error is large !!! \n");
		}
		cout << vo.points3D_Vec.size() << endl << endl;
		vo.points3D_Vec.clear();
		//printf("kfID: %d    frameID: %d \n", kfID, currentFrame.id);

		Mat show2;
		vector<KeyPoint> keypoints;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(nextFrame.pt, keypoints);
		drawKeypoints(nextFrame.img, keypoints, show2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(show2, currentFrame.pt, nextFrame.pt);

		putText(show2, //"FPS: " + to_string((1 / ((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)))) +
			"  ID: " + to_string(nextFrame.id) +
			"  R_err: " + to_string(r_errCurr) +
			"  t_err: " + to_string(t_errCurr) +
			"  Error_R: " + to_string(R_err) +
			"  Error_t: " + to_string(t_err), Point(5, currentFrame.img.rows - 40), 1, 1, Scalar(0, 255, 0), 2);
		imshow("test", show2);

		showTrajAndGTruth(traj, vo.T_messure, vo.T);

		lastFrame = currentFrame;
		currentFrame = nextFrame;
		findKeyframe = true;

		waitKey(1);
	}
}

void
odometryRun_ALL_PNP2()
{
	//--- set first frame ID number...
	frameNum = firstFrameID;

	//--- prepare VO for using...
	struct VisualOdometry vo;
	loadPoses(vo.poses_gt);
	trajectoryDistances(vo.poses_gt, vo.trajDistances);
	initializePose(firstFrameID, vo);
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);
	bool findKeyframe = false;
	int keyframeCounter = 0;
	int use = 0;
	double R_err = 0, t_err = 0, r_errCurr = 0, t_errCurr = 0;
	//-------------------------------------------------------------------

	//--- prepare for write pose file...
	string pose_gt_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_gt_Runtime.txt";
	string pose_em_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_em_Runtime.txt";
	ofstream out_gt(pose_gt_Runtime);
	ofstream out_em(pose_em_Runtime);

	for (int i = 0; i < vo.T.rows - 1; i++) {
		for (int j = 0; j< vo.T.cols; j++) {
			out_gt << vo.T.at<double>(i, j) << " ";
		}
	}
	out_gt << endl;

	for (int i = 0; i < vo.T_messure.rows - 1; i++) {
		for (int j = 0; j< vo.T_messure.cols; j++) {
			out_em << vo.T_messure.at<double>(i, j) << " ";
		}
	}
	out_em << endl;
	//--------------------------------------------------------------------

	//--- run odometry initial.
	struct Frame lastFrame;
	struct Frame currentFrame = odometryInitial(vo, lastFrame, vo.status);
	
	int kfID = lastFrame.id;
	vo.poses_messure.push_back(vo.T_messure.clone());
	lastFrame.P = vo.P_Keyframe.clone();
	int pts_size = countNonZero(vo.status);
	findKeyframe = (pts_size < pts_Size_KF_TH) ? true : false;

	//--- make projection matrix.
	currentFrame.P = makeProjectionMatrix(vo.T_messure);

	//--- truangulate points to 3D.
	vector<Point3d> pts3D;
	triangulate_Points(vo.P_Keyframe, currentFrame.P, vo.ptKF, currentFrame.pt, vo.status, pts3D);
	vo.points3D_Vec.push_back(pts3D);

	while (true) {
		//--- add next frame and prepare for using.
		struct Frame nextFrame;
		if (!addFrame(nextFrame)) {
			out_gt.close();
			out_em.close();
			waitKey(0);
			break;
		}

		if (findKeyframe) {
			findKeyframe = false;

			detectKeypoints(lastFrame.grayImg, lastFrame.pt);

			vo.ptSizeKF_TH = lastFrame.pt.size();
			vo.ptKF = lastFrame.pt;
			kfID = lastFrame.id;

			bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg,
											 lastFrame.pt, currentFrame.pt, vo.status, vo.ptSizeKF_TH);

			vo.status_after_tracking = vo.status;
			//pts_size = countNonZero(vo.status);
			//isKeyFrame = (pts_size < pts_Size_KF_TH) ? true : false;

			Mat T_f;
			bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.status, T_f, isKeyFrame);

			//--- truangulate points to 3D.
			vector<Point3d> pts3D;
			printf("Frame: %d => %d \n", lastFrame.id, currentFrame.id);
			triangulate_Points(lastFrame.P, currentFrame.P, vo.ptKF, currentFrame.pt, vo.status, pts3D);
			vo.points3D_Vec.push_back(pts3D);
		}

		//--- track kp in lbp frame and check if find key frame.
		vo.status = vo.status_after_tracking;
		bool isKeyFrame = trackKeypoints(currentFrame.grayImg, nextFrame.grayImg,
										 currentFrame.pt, nextFrame.pt, vo.status, vo.ptSizeKF_TH);

		//pts_size = countNonZero(vo.status);
		//isKeyFrame = (pts_size < countNonZero(vo.status_after_tracking) * pts_Size_KF_Ratio) ? true : false;

		//--- calculate rotation and translation, check if pose correct.
		Mat T_f;
		bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.status, T_f, isKeyFrame);

		Mat T_pnp = poseEstimatePnP(vo.points3D_Vec[0], nextFrame.pt, vo.status);

		//--- solve PnP for estimating camera pose.
		//Mat r_pnp, t_pnp, R_pnp;
		//solvePnP(vo.points3D_Vec[vo.points3D_Vec.size()-1], nextFrame.pt, K, Mat(), r_pnp, t_pnp, false, SOLVEPNP_ITERATIVE);
		//solvePnPRansac(vo.points3D_Vec[0], nextFrame.pt, K, Mat(), r_pnp, t_pnp, false, 1000, 1.0, 0.99);

		//--- inverse transformation from solvePnP.
		//Mat T_pnp = inverseTransformation(r_pnp, t_pnp);

		//--- make projection matrix using transformation matrix and save it to nextFrame.
		nextFrame.P = makeProjectionMatrix(T_pnp);

		//--- truangulate points to 3D.
		vector<Point3d> pts3D;
		printf("Frame: %d => %d \n", lastFrame.id, nextFrame.id);
		triangulate_Points(lastFrame.P, nextFrame.P, vo.ptKF, nextFrame.pt, vo.status, pts3D);
		vo.points3D_Vec.push_back(pts3D);

		if (isKeyFrame) {
			findKeyframe = true;
			keyframeCounter++;

			vector<Point3d> pt3D_Result;
			vector<Point2f> pt2D_Result;
			const int ptSize = vo.ptKF.size();
			const int vecSize = vo.points3D_Vec.size();
			for (int i = 0; i < ptSize; i++) {
				/*Point3d p(0, 0, 0);
				for (int j = 0; j < vecSize; j++) {
				p += voLBP.points3D_Vec[j].at(i);
				}
				p /= vecSize;

				bool push = false;
				int counter = 0;
				Point3d p_new(0, 0, 0);
				for (int j = 0; j < vecSize; j++) {
				double deltaX = abs( (p - voLBP.points3D_Vec[j].at(i)).x );
				double deltaY = abs( (p - voLBP.points3D_Vec[j].at(i)).y );

				if (deltaX <= 0.1 && deltaY <= 0.1) {
				push = true;
				p_new += voLBP.points3D_Vec[j].at(i);
				counter++;
				}
				}
				if (push) {
				p_new /= counter;
				pt3D_Result.push_back(p_new);
				pt2D_Result.push_back(nextFrame.lbp_feature.lbpPt.at(i));
				}*/

				Point3d mu(0, 0, 0);
				int nonZeroPt3DNum = 0;
				for (int j = 0; j < vecSize; j++) {
					Point3d pt3D = vo.points3D_Vec[j].at(i);
					if ( pt3D != Point3d(0, 0, 0) ) {
						mu += pt3D;
						nonZeroPt3DNum++;
					}
					//mu += vo.points3D_Vec[j].at(i);
				}
				//mu /= vecSize;

				if (nonZeroPt3DNum == 0) {
					pt3D_Result.push_back(Point3d(0, 0, 0));
					pt2D_Result.push_back(nextFrame.pt.at(i));
					continue;
				}

				mu /= nonZeroPt3DNum;
					

				Point3d Sigma(0, 0, 0);
				vector<Point3d> deltaVec;
				for (int j = 0; j < vecSize; j++) {
					const double deltaX = abs((mu - vo.points3D_Vec[j].at(i)).x);
					const double deltaY = abs((mu - vo.points3D_Vec[j].at(i)).y);
					const double deltaZ = abs((mu - vo.points3D_Vec[j].at(i)).z);
					deltaVec.push_back(Point3d(deltaX, deltaY, deltaZ));

					Sigma.x += pow(deltaX, 2);
					Sigma.y += pow(deltaY, 2);
					Sigma.z += pow(deltaZ, 2);
				}
				//Sigma /= vecSize;
				Sigma /= nonZeroPt3DNum;
				Sigma = Point3d(sqrtf(Sigma.x), sqrtf(Sigma.y), sqrtf(Sigma.z));

				bool push = false;
				int counter = 0;
				Point3d p_new(0, 0, 0);
				for (int j = 0; j < vecSize; j++) {
					Point3d delta = deltaVec[j];
					const double errScoreX = delta.x / Sigma.x;
					const double errScoreY = delta.y / Sigma.y;
					const double errScoreZ = delta.z / Sigma.z;

					if (errScoreX < 1.2 && errScoreY < 1.2 && errScoreZ < 1.2) {
						push = true;
						p_new += vo.points3D_Vec[j].at(i);
						counter++;
					}
				}

				if (push) {
					p_new /= counter;
					pt3D_Result.push_back(p_new);
					pt2D_Result.push_back(nextFrame.pt.at(i));
				}
				else {
					//printf("error too large, use mu !!! \n\n");
					//pt3D_Result.push_back(mu);
					pt3D_Result.push_back(Point3d(0, 0, 0));
					pt2D_Result.push_back(nextFrame.pt.at(i));
				}
			}
			//voLBP.points3D_Vec.clear();
			if (pt3D_Result.size() <= 4) {
				cout << "no pass !!!" << endl;
			}

			Mat T_pnp = poseEstimatePnP(pt3D_Result, pt2D_Result, vo.status);

			//Mat r_pnp, t_pnp, R_pnp;
			//solvePnP(vo.points3D_Vec[vecSize-1], nextFrame.pt, K, Mat(), r_pnp, t_pnp, false, SOLVEPNP_ITERATIVE);
			//solvePnPRansac(vo.points3D_Vec[vecSize - 1], nextFrame.pt, K, Mat(), r_pnp, t_pnp, false, 1000, 1.0, 0.99);
			//solvePnPRansac(pt3D_Result, pt2D_Result, K, Mat(), r_pnp, t_pnp, false, 1000, 1.0, 0.99);

			//--- inverse transformation from solvePnP.
			//Mat T_pnp = inverseTransformation(r_pnp, t_pnp);

			//--- make projection matrix using transformation matrix.
			Mat P_final = makeProjectionMatrix(T_pnp);
			nextFrame.P = P_final.clone();
			vo.P_Keyframe = P_final.clone();

			//--- save pose data.
			vo.T = vo.poses_gt[currentFrame.id];
			vo.T_messure = T_pnp.clone();
			vo.poses_messure.push_back(T_pnp.clone());

			//--- calc. error.
			CalcSequenceErrors(vo.poses_gt, vo.poses_messure, vo.err, kfID, currentFrame.id, scale);
			use = vo.err.size() - 1;
			r_errCurr = vo.err[use].r_err / scale;
			t_errCurr = vo.err[use].t_err / scale;
			GetStats(vo.err, t_err, R_err);

			//--- write file.
			for (int i = 0; i < vo.T.rows - 1; i++) {
				for (int j = 0; j < vo.T.cols; j++) {
					out_gt << vo.T.at<double>(i, j) << " ";
				}
			}
			out_gt << endl;

			for (int i = 0; i < vo.T_messure.rows - 1; i++) {
				for (int j = 0; j < vo.T_messure.cols; j++) {
					out_em << vo.T_messure.at<double>(i, j) << " ";
				}
			}
			out_em << endl;


			if (r_errCurr >= 0.005) {
				//printf("Rotation error is large !!! \n");
			}
			cout << vo.points3D_Vec.size() << endl << endl;
			vo.points3D_Vec.clear();
			//printf("kfID: %d    frameID: %d \n", kfID, currentFrame.id);
		}

		Mat show2;
		vector<KeyPoint> keypoints;
		KeyPoint::convert(nextFrame.pt, keypoints);
		drawKeypoints(nextFrame.img, keypoints, show2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		drawOpticalFlow(show2, currentFrame.pt, nextFrame.pt);

		putText(show2, //"FPS: " + to_string((1 / ((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)))) +
			"  ID: " + to_string(nextFrame.id) +
			"  R_err: " + to_string(r_errCurr) +
			"  t_err: " + to_string(t_errCurr) +
			"  Error_R: " + to_string(R_err) +
			"  Error_t: " + to_string(t_err), Point(5, currentFrame.img.rows - 40), 1, 1, Scalar(0, 255, 0), 2);
		imshow("test", show2);

		showTrajAndGTruth(traj, vo.T_messure, vo.T);

		if (isKeyFrame) {
			lastFrame = currentFrame;
			currentFrame = nextFrame;
		}
		
		waitKey(1);
	}
}

void
odometryRun_KF()
{
	//--- set first frame ID number...
	frameNum = firstFrameID;

	//--- prepare VO for using...
	struct VisualOdometry vo;
	loadPoses(vo.poses_gt);
	trajectoryDistances(vo.poses_gt, vo.trajDistances);
	initializePose(firstFrameID, vo);
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);
	bool findKeyframe = false;
	int keyframeCounter = 0;
	int use = 0;
	double R_err = 0, t_err = 0, r_errCurr = 0, t_errCurr = 0;

	//--- prepare for write pose file...
	string pose_gt_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_gt_Runtime.txt";
	string pose_em_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_em_Runtime.txt";
	ofstream out_gt(pose_gt_Runtime);
	ofstream out_em(pose_em_Runtime);

	for (int i = 0; i < vo.T.rows - 1; i++) {
		for (int j = 0; j< vo.T.cols; j++) {
			out_gt << vo.T.at<double>(i, j) << " ";
		}
	}
	out_gt << endl;

	for (int i = 0; i < vo.T_messure.rows - 1; i++) {
		for (int j = 0; j< vo.T_messure.cols; j++) {
			out_em << vo.T_messure.at<double>(i, j) << " ";
		}
	}
	out_em << endl;

	//--- prepare first frame...
	struct Frame lastFrame;
	if (!addFrame(lastFrame))
		return;
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);

	vo.ptSizeKF_TH = lastFrame.pt.size();
	vo.ptKF = lastFrame.pt;
	int kfID = lastFrame.id;

	bool have_3DPt = false;
	while (true) {
		//--- add next frame and prepare for using...
		struct Frame currentFrame;
		if (!addFrame(currentFrame)) {
			out_gt.close();
			out_em.close();
			waitKey(0);
			break;
		}

		if (findKeyframe) {
			findKeyframe = false;

			detectKeypoints(lastFrame.grayImg, lastFrame.pt);


			vo.ptSizeKF_TH = lastFrame.pt.size();
			vo.ptKF = lastFrame.pt;
			kfID = lastFrame.id;
		}

		//--- track kp in lbp frame and check if find key frame...
		bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg,
										 lastFrame.pt, currentFrame.pt, vo.status, vo.ptSizeKF_TH);

		vo.status_after_tracking = vo.status;

		/*Mat showTest;
		vector<KeyPoint> keypointsTest;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.lbp_feature.lbpPt, keypointsTest);
		drawKeypoints(currentFrame.img, keypointsTest, showTest, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(showTest, lastFrame.lbp_feature.lbpPt, currentFrame.lbp_feature.lbpPt);
		imshow("test", showTest);*/
		//waitKey(0);

		//--- calculate rotation and translation, check if pose correct...
		Mat T_f;
		bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.status, T_f, isKeyFrame);
		//bool isCorrect = poseEstimate(voLBP.ptKF, currentFrame.lbp_feature.lbpPt, lastFrame.lbp_feature.lbpPt, voLBP.points3D_Vec, T_f, have_3DPt);

		/*Mat showTest2;
		vector<KeyPoint> keypointsTest2;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.lbp_feature.lbpPt, keypointsTest2);
		drawKeypoints(currentFrame.img, keypointsTest2, showTest2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(showTest2, lastFrame.lbp_feature.lbpPt, currentFrame.lbp_feature.lbpPt);
		imshow("test", showTest2);*/

		/*vector<errors> err_f;
		double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[kfID];
		Mat T_f_temp = T_f.clone();
		Mat t = T_f_temp(Range(0, 3), Range(3, 4));
		t *= scale;
		poseFrameErrors(vo.poses_gt, T_f_temp, kfID, currentFrame.id, err_f);
		printf("R: %f   t: %f \n", err_f[0].r_err, err_f[0].t_err);*/


		if (isCorrect) {
			have_3DPt = true;

			double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[kfID];
			Mat T = vo.T_messure.clone();
			updataPose(scale, T_f, T);

			//--- make projection matrix
			Mat P = makeProjectionMatrix(T);
			//voLBP.P_Vec.push_back(P.clone());

			//Mat points4D;
			//triangulatePoints(voLBP.P_Vec[keyframeCounter], P, voLBP.ptKF, currentFrame.lbp_feature.lbpPt, points4D);
			//triangulatePoints(vo.P_Keyframe, P, vo.ptKF, currentFrame.pt, points4D);
			//triangulatePoints(voLBP.P_Vec[currentFrame.id], voLBP.P_Vec[kfID], currentFrame.lbp_feature.lbpPt, voLBP.ptKF, points4D);

			//--- normalize 4D points to 3D points and save to pt3DVec...
			//vector<Point3d> points3D;
			//normalizePoints(points4D, points3D);
			//vo.points3D_Vec.push_back(points3D);

			vector<Point3d> points3D;
			printf("Frame: %d => %d \n", lastFrame.id, currentFrame.id);
			triangulate_Points(vo.P_Keyframe, P, vo.ptKF, currentFrame.pt, vo.status, points3D);
			vo.points3D_Vec.push_back(points3D);

			//showTriangulation(voLBP.ptKF, currentFrame.lbp_feature.lbpPt, R, t, points3D);

			if (isKeyFrame) {
				findKeyframe = true;
				have_3DPt = false;
				keyframeCounter++;

				vector<Point3d> pt3D_Result;
				vector<Point2f> pt2D_Result;
				const int ptSize = vo.ptKF.size();
				const int vecSize = vo.points3D_Vec.size();
				for (int i = 0; i < ptSize; i++) {
					/*Point3d p(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
					p += voLBP.points3D_Vec[j].at(i);
					}
					p /= vecSize;

					bool push = false;
					int counter = 0;
					Point3d p_new(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
					double deltaX = abs( (p - voLBP.points3D_Vec[j].at(i)).x );
					double deltaY = abs( (p - voLBP.points3D_Vec[j].at(i)).y );

					if (deltaX <= 0.1 && deltaY <= 0.1) {
					push = true;
					p_new += voLBP.points3D_Vec[j].at(i);
					counter++;
					}
					}
					if (push) {
					p_new /= counter;
					pt3D_Result.push_back(p_new);
					pt2D_Result.push_back(currentFrame.lbp_feature.lbpPt.at(i));
					}*/

					Point3d mu(0, 0, 0);
					int nonZeroPt3DNum = 0;
					for (int j = 0; j < vecSize; j++) {
						Point3d pt3D = vo.points3D_Vec[j].at(i);
						if (pt3D != Point3d(0, 0, 0)) {
							mu += pt3D;
							nonZeroPt3DNum++;
						}
						//mu += vo.points3D_Vec[j].at(i);
					}

					if (nonZeroPt3DNum == 0) {
						pt3D_Result.push_back(Point3d(0, 0, 0));
						pt2D_Result.push_back(currentFrame.pt.at(i));
						continue;
					}

					//mu /= vecSize;
					mu /= nonZeroPt3DNum;

					Point3d Sigma(0, 0, 0);
					vector<Point3d> deltaVec;
					for (int j = 0; j < vecSize; j++) {
						const double deltaX = abs((mu - vo.points3D_Vec[j].at(i)).x);
						const double deltaY = abs((mu - vo.points3D_Vec[j].at(i)).y);
						const double deltaZ = abs((mu - vo.points3D_Vec[j].at(i)).z);
						deltaVec.push_back(Point3d(deltaX, deltaY, deltaZ));

						Sigma.x += pow(deltaX, 2);
						Sigma.y += pow(deltaY, 2);
						Sigma.z += pow(deltaZ, 2);
					}
					//Sigma /= vecSize;
					Sigma /= nonZeroPt3DNum;
					Sigma = Point3d(sqrtf(Sigma.x), sqrtf(Sigma.y), sqrtf(Sigma.z));

					bool push = false;
					int counter = 0;
					Point3d p_new(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
						Point3d delta = deltaVec[j];
						const double errScoreX = delta.x / Sigma.x;
						const double errScoreY = delta.y / Sigma.y;
						const double errScoreZ = delta.z / Sigma.z;

						if (errScoreX < 1.2 && errScoreY < 1.2 && errScoreZ < 1.2) {
							push = true;
							p_new += vo.points3D_Vec[j].at(i);
							counter++;
						}
					}

					if (push) {
						p_new /= counter;
						pt3D_Result.push_back(p_new);
						pt2D_Result.push_back(currentFrame.pt.at(i));
					}
					else {
						//printf("error too large, use mu !!! \n\n");
						//pt3D_Result.push_back(mu);
						pt3D_Result.push_back(Point3d(0, 0, 0));
						pt2D_Result.push_back(currentFrame.pt.at(i));
					}
				}
				//voLBP.points3D_Vec.clear();
				if (pt3D_Result.size() <= 4) {
					cout << "no pass !!!" << endl;
				}

				//Mat r_pnp, t_pnp, R_pnp;
				//solvePnP(vo.points3D_Vec[vecSize-1], currentFrame.pt, K, Mat(), r_pnp, t_pnp, false, SOLVEPNP_ITERATIVE);
				//solvePnPRansac(pt3D_Result, currentFrame.lbp_feature.lbpPt, K, Mat(), r_pnp, t_pnp, false, 1000, 1, 0.99);
				//solvePnPRansac(pt3D_Result, pt2D_Result, K, Mat(), r_pnp, t_pnp, false, 5000, 1.0, 0.995);
				//solvePnPRansac(vo.points3D_Vec[vecSize-1], currentFrame.pt, K, Mat(), r_pnp, t_pnp, false, 1000, 1.0, 0.99);

				//--- inverse transformation from solvePnP...
				//Mat T_pnp = inverseTransformation(r_pnp, t_pnp);

				Mat T_pnp = poseEstimatePnP(pt3D_Result, pt2D_Result, vo.status);

				//--- make projection matrix using transformation matrix...
				Mat P_final = makeProjectionMatrix(T_pnp);
				//voLBP.P_Vec.push_back(P_final.clone());
				vo.P_Keyframe = P_final.clone();

				//updataPose(scale, voLBP, T_pnp);
				//voLBP.poses_messure.push_back(voLBP.T_messure.clone());
				vo.T = vo.poses_gt[currentFrame.id];
				vo.T_messure = T_pnp.clone();
				vo.poses_messure.push_back(T_pnp.clone());

				CalcSequenceErrors(vo.poses_gt, vo.poses_messure, vo.err, kfID, currentFrame.id, scale);
				use = vo.err.size() - 1;
				r_errCurr = vo.err[use].r_err / scale;
				t_errCurr = vo.err[use].t_err / scale;
				GetStats(vo.err, t_err, R_err);

				for (int i = 0; i < vo.T.rows - 1; i++) {
					for (int j = 0; j< vo.T.cols; j++) {
						out_gt << vo.T.at<double>(i, j) << " ";
					}
				}
				out_gt << endl;

				for (int i = 0; i < vo.T_messure.rows - 1; i++) {
					for (int j = 0; j< vo.T_messure.cols; j++) {
						out_em << vo.T_messure.at<double>(i, j) << " ";
					}
				}
				out_em << endl;

				if (r_errCurr >= 0.005) {
					//printf("Rotation error is large !!! \n");
				}
				cout << vo.points3D_Vec.size() << endl << endl;
				vo.points3D_Vec.clear();
				//printf("kfID: %d    frameID: %d \n", kfID, currentFrame.id);
			}
		}
		else {
			//cout << "next frame !!! \n" << endl;
		}
		//bundleAdjustment(points3D, currentFrame.lbp_feature.lbpPt, K, R, t);

		Mat show2;
		vector<KeyPoint> keypoints;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.pt, keypoints);
		drawKeypoints(currentFrame.img, keypoints, show2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(show2, lastFrame.pt, currentFrame.pt);

		putText(show2, //"FPS: " + to_string((1 / ((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)))) +
			"  ID: " + to_string(currentFrame.id) +
			"  R_err: " + to_string(r_errCurr) +
			"  t_err: " + to_string(t_errCurr) +
			"  Error_R: " + to_string(R_err) +
			"  Error_t: " + to_string(t_err), Point(5, lastFrame.img.rows - 40), 1, 1, Scalar(0, 255, 0), 2);
		imshow("test", show2);

		showTrajAndGTruth(traj, vo.T_messure, vo.T);

		lastFrame = currentFrame;

		waitKey(1);
	}
}

void
odometryRun_PNP_Test()
{
	//--- set first frame ID number...
	frameNum = firstFrameID;

	//--- prepare VO for using...
	struct VisualOdometry vo;
	loadPoses(vo.poses_gt);
	trajectoryDistances(vo.poses_gt, vo.trajDistances);
	initializePose(firstFrameID, vo);
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);
	bool findKeyframe = false;
	int keyframeCounter = 0;
	int use = 0;
	double R_err = 0, t_err = 0, r_errCurr = 0, t_errCurr = 0;

	//--- prepare for write pose file...
	string pose_gt_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_gt_Runtime.txt";
	string pose_em_Runtime = "D:\\Document\\NCTU\\論文進度報告\\VisualOdometry - (MSB_LBP)\\pose_em_Runtime.txt";
	ofstream out_gt(pose_gt_Runtime);
	ofstream out_em(pose_em_Runtime);

	for (int i = 0; i < vo.T.rows - 1; i++) {
		for (int j = 0; j< vo.T.cols; j++) {
			out_gt << vo.T.at<double>(i, j) << " ";
		}
	}
	out_gt << endl;

	for (int i = 0; i < vo.T_messure.rows - 1; i++) {
		for (int j = 0; j< vo.T_messure.cols; j++) {
			out_em << vo.T_messure.at<double>(i, j) << " ";
		}
	}
	out_em << endl;

	//--- prepare first frame...
	struct Frame lastFrame;
	if (!addFrame(lastFrame))
		return;
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);

	vo.ptSizeKF_TH = lastFrame.pt.size();
	vo.ptKF = lastFrame.pt;
	int kfID = lastFrame.id;

	bool have_3DPt = false;
	while (true) {
		//--- add next frame and prepare for using...
		struct Frame currentFrame;
		if (!addFrame(currentFrame)) {
			out_gt.close();
			out_em.close();
			waitKey(0);
			break;
		}

		if (currentFrame.id == 654) {
			bool test = true;
		}

		if (findKeyframe) {
			findKeyframe = false;

			detectKeypoints(lastFrame.grayImg, lastFrame.pt);


			vo.ptSizeKF_TH = lastFrame.pt.size();
			vo.ptKF = lastFrame.pt;
			kfID = lastFrame.id;
		}

		//--- track kp in lbp frame and check if find key frame...
		bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg,
			lastFrame.pt, currentFrame.pt,
			vo.ptKF, vo.ptSizeKF_TH, vo.points3D_Vec, have_3DPt);

		/*Mat showTest;
		vector<KeyPoint> keypointsTest;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.lbp_feature.lbpPt, keypointsTest);
		drawKeypoints(currentFrame.img, keypointsTest, showTest, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(showTest, lastFrame.lbp_feature.lbpPt, currentFrame.lbp_feature.lbpPt);
		imshow("test", showTest);*/
		//waitKey(0);

		//--- calculate rotation and translation, check if pose correct...
		Mat T_f;
		bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.ptKF, vo.points3D_Vec, T_f, have_3DPt);
		//bool isCorrect = poseEstimate(voLBP.ptKF, currentFrame.lbp_feature.lbpPt, lastFrame.lbp_feature.lbpPt, voLBP.points3D_Vec, T_f, have_3DPt);

		/*Mat showTest2;
		vector<KeyPoint> keypointsTest2;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.lbp_feature.lbpPt, keypointsTest2);
		drawKeypoints(currentFrame.img, keypointsTest2, showTest2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(showTest2, lastFrame.lbp_feature.lbpPt, currentFrame.lbp_feature.lbpPt);
		imshow("test", showTest2);*/

		/*vector<errors> err_f;
		double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[kfID];
		Mat T_f_temp = T_f.clone();
		Mat t = T_f_temp(Range(0, 3), Range(3, 4));
		t *= scale;
		poseFrameErrors(vo.poses_gt, T_f_temp, kfID, currentFrame.id, err_f);
		printf("R: %f   t: %f \n", err_f[0].r_err, err_f[0].t_err);*/


		if (isCorrect) {
			//have_3DPt = true;

			//double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[kfID];
			//Mat T = vo.T_messure.clone();
			//updataPose(scale, T_f, T);

			//--- make projection matrix
			//Mat P = makeProjectionMatrix(T);
			//voLBP.P_Vec.push_back(P.clone());

			//Mat points4D;
			//triangulatePoints(voLBP.P_Vec[keyframeCounter], P, voLBP.ptKF, currentFrame.lbp_feature.lbpPt, points4D);
			//triangulatePoints(vo.P_Keyframe, P, vo.ptKF, currentFrame.pt, points4D);
			//triangulatePoints(voLBP.P_Vec[currentFrame.id], voLBP.P_Vec[kfID], currentFrame.lbp_feature.lbpPt, voLBP.ptKF, points4D);

			//--- normalize 4D points to 3D points and save to pt3DVec...
			//vector<Point3d> points3D;
			//normalizePoints(points4D, points3D);
			//vo.points3D_Vec.push_back(points3D);

			//--- testing
			// Chierality check
			//vector<uchar> status;
			//for (unsigned int i = 0; i < points3D.size(); i++)
			//status.push_back((points3D[i].z > 0) ? 1 : 0);

			//cout << "triangulation likes " << countNonZero(Mat(status))
			//<< " out of " << currentFrame.pt.size()
			//<< " (" << (float)(countNonZero(Mat(status))) / (float)(currentFrame.pt.size()) * 100 << "%)" << endl << endl;

			//showTriangulation(voLBP.ptKF, currentFrame.lbp_feature.lbpPt, R, t, points3D);

			if (isKeyFrame) {
				findKeyframe = true;
				//have_3DPt = false;
				keyframeCounter++;

				//vector<Point3d> pt3D_Result;
				//vector<Point2f> pt2D_Result;
				//const int ptSize = vo.ptKF.size();
				//const int vecSize = vo.points3D_Vec.size();
				//for (int i = 0; i < ptSize; i++) {
					/*Point3d p(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
					p += voLBP.points3D_Vec[j].at(i);
					}
					p /= vecSize;

					bool push = false;
					int counter = 0;
					Point3d p_new(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
					double deltaX = abs( (p - voLBP.points3D_Vec[j].at(i)).x );
					double deltaY = abs( (p - voLBP.points3D_Vec[j].at(i)).y );

					if (deltaX <= 0.1 && deltaY <= 0.1) {
					push = true;
					p_new += voLBP.points3D_Vec[j].at(i);
					counter++;
					}
					}
					if (push) {
					p_new /= counter;
					pt3D_Result.push_back(p_new);
					pt2D_Result.push_back(currentFrame.lbp_feature.lbpPt.at(i));
					}*/

					/*Point3d mu(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
						mu += vo.points3D_Vec[j].at(i);
					}
					mu /= vecSize;

					Point3d Sigma(0, 0, 0);
					vector<Point3d> deltaVec;
					for (int j = 0; j < vecSize; j++) {
						const double deltaX = abs((mu - vo.points3D_Vec[j].at(i)).x);
						const double deltaY = abs((mu - vo.points3D_Vec[j].at(i)).y);
						const double deltaZ = abs((mu - vo.points3D_Vec[j].at(i)).z);
						deltaVec.push_back(Point3d(deltaX, deltaY, deltaZ));

						Sigma.x += pow(deltaX, 2);
						Sigma.y += pow(deltaY, 2);
						Sigma.z += pow(deltaZ, 2);
					}
					Sigma /= vecSize;
					Sigma = Point3d(sqrtf(Sigma.x), sqrtf(Sigma.y), sqrtf(Sigma.z));

					bool push = false;
					int counter = 0;
					Point3d p_new(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
						Point3d delta = deltaVec[j];
						const double errScoreX = delta.x / Sigma.x;
						const double errScoreY = delta.y / Sigma.y;
						const double errScoreZ = delta.z / Sigma.z;

						if (errScoreX < 1.2 && errScoreY < 1.2 && errScoreZ < 1.2) {
							push = true;
							p_new += vo.points3D_Vec[j].at(i);
							counter++;
						}
					}

					if (push) {
						p_new /= counter;
						pt3D_Result.push_back(p_new);
						pt2D_Result.push_back(currentFrame.pt.at(i));
					}
					else {
						//printf("error too large, use mu !!! \n\n");
						pt3D_Result.push_back(mu);
						pt2D_Result.push_back(currentFrame.pt.at(i));
					}
				}
				//voLBP.points3D_Vec.clear();
				if (pt3D_Result.size() <= 4) {
					cout << "no pass !!!" << endl;
				}

				Mat r_pnp, t_pnp, R_pnp;
				//solvePnP(vo.points3D_Vec[vecSize-1], currentFrame.pt, K, Mat(), r_pnp, t_pnp, false, SOLVEPNP_ITERATIVE);
				//solvePnPRansac(pt3D_Result, currentFrame.lbp_feature.lbpPt, K, Mat(), r_pnp, t_pnp, false, 1000, 1, 0.99);
				//solvePnPRansac(pt3D_Result, pt2D_Result, K, Mat(), r_pnp, t_pnp, false, 5000, 1.0, 0.995);
				solvePnPRansac(vo.points3D_Vec[vecSize - 1], currentFrame.pt, K, Mat(), r_pnp, t_pnp, false, 1000, 1.0, 0.99);

				//--- inverse transformation from solvePnP...
				Mat T_pnp = inverseTransformation(r_pnp, t_pnp);

				//--- make projection matrix using transformation matrix...
				Mat P_final = makeProjectionMatrix(T_pnp);
				//voLBP.P_Vec.push_back(P_final.clone());
				vo.P_Keyframe = P_final.clone();*/

				//updataPose(scale, voLBP, T_pnp);
				//voLBP.poses_messure.push_back(voLBP.T_messure.clone());
				
				//--- for testing...
				double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[kfID];
				Mat T = vo.T_messure.clone();
				updataPose(scale, T_f, T);

				vo.T = vo.poses_gt[currentFrame.id];
				vo.T_messure = T.clone();
				vo.poses_messure.push_back(T.clone());

				CalcSequenceErrors(vo.poses_gt, vo.poses_messure, vo.err, kfID, currentFrame.id, scale);
				use = vo.err.size() - 1;
				r_errCurr = vo.err[use].r_err / scale;
				t_errCurr = vo.err[use].t_err / scale;
				GetStats(vo.err, t_err, R_err);

				for (int i = 0; i < vo.T.rows - 1; i++) {
					for (int j = 0; j< vo.T.cols; j++) {
						out_gt << vo.T.at<double>(i, j) << " ";
					}
				}
				out_gt << endl;

				for (int i = 0; i < vo.T_messure.rows - 1; i++) {
					for (int j = 0; j< vo.T_messure.cols; j++) {
						out_em << vo.T_messure.at<double>(i, j) << " ";
					}
				}
				out_em << endl;

				if (r_errCurr >= 0.005) {
					//printf("Rotation error is large !!! \n");
				}
				cout << vo.points3D_Vec.size() << endl << endl;
				vo.points3D_Vec.clear();
				//printf("kfID: %d    frameID: %d \n", kfID, currentFrame.id);
			}
		}
		else {
			//cout << "next frame !!! \n" << endl;
		}
		//bundleAdjustment(points3D, currentFrame.lbp_feature.lbpPt, K, R, t);

		Mat show2;
		vector<KeyPoint> keypoints;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(currentFrame.pt, keypoints);
		drawKeypoints(currentFrame.img, keypoints, show2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(show2, lastFrame.pt, currentFrame.pt);

		putText(show2, //"FPS: " + to_string((1 / ((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)))) +
			"  ID: " + to_string(currentFrame.id) +
			"  R_err: " + to_string(r_errCurr) +
			"  t_err: " + to_string(t_errCurr) +
			"  Error_R: " + to_string(R_err) +
			"  Error_t: " + to_string(t_err), Point(5, lastFrame.img.rows - 40), 1, 1, Scalar(0, 255, 0), 2);
		imshow("test", show2);

		showTrajAndGTruth(traj, vo.T_messure, vo.T);

		lastFrame = currentFrame;

		waitKey(1);
	}
}

void
odometryRunLBP_PNP2()
{
	//--- set first frame ID number...
	frameNum = firstFrameID;

	//--- prepare VO for using...
	struct VisualOdometry vo;
	loadPoses(vo.poses_gt);
	trajectoryDistances(vo.poses_gt, vo.trajDistances);
	initializePose(firstFrameID, vo);
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);
	bool findKeyframe = false;
	int keyframeCounter = 0;
	int use = 0;
	double R_err = 0, t_err = 0, r_errCurr = 0, t_errCurr = 0;

	//--- prepare first frame...
	struct Frame lastFrame;
	if (!addFrame(lastFrame))
		return;
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);
	vo.ptSizeKF_TH = lastFrame.pt.size();
	vo.ptKF = lastFrame.pt;
	int kfID = lastFrame.id;
	lastFrame.T = vo.poses_gt.at(firstFrameID).clone();
	lastFrame.P = vo.P_Vec.at(0).clone();

	bool have_3DPt = false;
	while (true) {
		//--- add next frame and prepare for using...
		struct Frame currentFrame;
		if (!addFrame(currentFrame)) {
			waitKey(0);
			break;
		}

		if (findKeyframe) {
			findKeyframe = false;

			//lastFrame.lbp_feature.lbpFrame = lastFrame.grayImg;
			//goodFeaturesToTrack(lastFrame.lbp_feature.lbpFrame, lastFrame.lbp_feature.lbpPt, 3000, 0.01, 20, Mat(), 5, true, 0.06);
			detectKeypoints(lastFrame.grayImg, lastFrame.pt);
			vo.ptSizeKF_TH = lastFrame.pt.size();
			vo.ptKF = lastFrame.pt;
			kfID = lastFrame.id;
		}

		//--- track kp in lbp frame and check if find key frame...
		bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg,
			lastFrame.pt, currentFrame.pt,
			vo.ptKF, vo.ptSizeKF_TH, vo.points3D_Vec, have_3DPt);

		//--- calculate rotation and translation, check if pose correct...
		Mat T_f;
		bool isCorrect = poseEstimate(vo.ptKF, currentFrame.pt, lastFrame.pt, vo.points3D_Vec, T_f, have_3DPt);

		/*vector<errors> err_f;
		double scale = voLBP.trajDistances[currentFrame.id] - voLBP.trajDistances[lastFrame.id];
		Mat T_f_temp = T_f.clone();
		Mat t = T_f_temp(Range(0, 3), Range(3, 4));
		t *= scale;
		poseFrameErrors(voLBP.poses_gt, T_f_temp, lastFrame.id, currentFrame.id, err_f);
		printf("R: %f   t: %f \n", err_f[0].r_err, err_f[0].t_err);*/

		if (isCorrect) {
			have_3DPt = true;

			double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[lastFrame.id];
			Mat T = lastFrame.T.clone();
			updataPose(scale, T_f, T);
			currentFrame.T = T.clone();

			//--- make projection matrix
			Mat P = makeProjectionMatrix(T);
			currentFrame.P = P.clone();

			Mat points4D;
			triangulatePoints(lastFrame.P, currentFrame.P, lastFrame.pt, currentFrame.pt, points4D);

			//--- normalize 4D points to 3D points and save to pt3DVec...
			vector<Point3d> points3D;
			normalizePoints(points4D, points3D);
			vo.points3D_Vec.push_back(points3D);

			//Mat R = T(Range(0, 3), Range(0, 3));
			//Mat t = T(Range(0, 3), Range(3, 4));
			//showTriangulation(lastFrame.pt, currentFrame.pt, R, t, points3D);

			if (isKeyFrame) {
				findKeyframe = true;
				have_3DPt = false;
				keyframeCounter++;

				vector<Point3d> pt3D_Result;
				vector<Point2f> pt2D_Result;
				const int ptSize = vo.ptKF.size();
				const int vecSize = vo.points3D_Vec.size();
				for (int i = 0; i < ptSize; i++) {
					Point3d mu(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
						mu += vo.points3D_Vec[j].at(i);
					}
					mu /= vecSize;

					Point3d Sigma(0, 0, 0);
					vector<Point3d> deltaVec;
					for (int j = 0; j < vecSize; j++) {
						const double deltaX = abs((mu - vo.points3D_Vec[j].at(i)).x);
						const double deltaY = abs((mu - vo.points3D_Vec[j].at(i)).y);
						const double deltaZ = abs((mu - vo.points3D_Vec[j].at(i)).z);
						deltaVec.push_back(Point3d(deltaX, deltaY, deltaZ));

						Sigma.x += pow(deltaX, 2);
						Sigma.y += pow(deltaY, 2);
						Sigma.z += pow(deltaZ, 2);
					}
					Sigma /= vecSize;
					Sigma = Point3d( sqrtf(Sigma.x), sqrtf(Sigma.y), sqrtf(Sigma.z) );

					bool push = false;
					int counter = 0;
					Point3d p_new(0, 0, 0);
					for (int j = 0; j < vecSize; j++) {
						Point3d delta = deltaVec[j];
						const double errScoreX = delta.x / Sigma.x;
						const double errScoreY = delta.y / Sigma.y;
						const double errScoreZ = delta.z / Sigma.z;

						if ( errScoreX < 1.2 && errScoreY < 1.2 && errScoreZ < 1.2 ) {
							push = true;
							p_new += vo.points3D_Vec[j].at(i);
							counter++;
						}
					}

					if (push) {
						p_new /= counter;
						pt3D_Result.push_back(p_new);
						pt2D_Result.push_back(currentFrame.pt.at(i));
					}
					else {
						//printf("error too large, use mu !!! \n\n");
						pt3D_Result.push_back(mu);
						pt2D_Result.push_back(currentFrame.pt.at(i));
					}
				}
				//voLBP.points3D_Vec.clear();

				Mat r_pnp, t_pnp;
				solvePnPRansac(pt3D_Result, pt2D_Result, K, Mat(), r_pnp, t_pnp, false, 5000, 1.0, 0.99);

				//--- inverse transformation from solvePnP...
				Mat T_pnp = inverseTransformation(r_pnp, t_pnp);
				currentFrame.T = T_pnp.clone();

				//--- make projection matrix using transformation matrix...
				Mat P_final = makeProjectionMatrix(T_pnp);
				currentFrame.P = P_final.clone();
				//voLBP.P_Vec.push_back(P_final.clone());
				vo.P_Keyframe = P_final.clone();

				//updataPose(scale, voLBP, T_pnp);
				//voLBP.poses_messure.push_back(voLBP.T_messure.clone());
				vo.T = vo.poses_gt[currentFrame.id];
				vo.T_messure = T_pnp.clone();
				vo.poses_messure.push_back(T_pnp.clone());

				CalcSequenceErrors(vo.poses_gt, vo.poses_messure, vo.err, kfID, currentFrame.id, scale);
				use = vo.err.size() - 1;
				r_errCurr = vo.err[use].r_err;
				t_errCurr = vo.err[use].t_err;
				GetStats(vo.err, t_err, R_err);

				if (r_errCurr >= 0.005) {
					printf("Rotation error is large !!! \n");
				}
				cout << vo.points3D_Vec.size() << endl << endl;
				vo.points3D_Vec.clear();
				//printf("kfID: %d    frameID: %d \n", kfID, currentFrame.id);
			}
		}
		else {
			//cout << "next frame !!! \n" << endl;
		}
		//bundleAdjustment(points3D, currentFrame.lbp_feature.lbpPt, K, R, t);

		Mat show2;
		vector<KeyPoint> keypoints;
		//KeyPoint::convert(lastFrame.pt, keypoints);
		KeyPoint::convert(lastFrame.pt, keypoints);
		drawKeypoints(currentFrame.img, keypoints, show2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//drawOpticalFlow(show, lastFrame.pt, currentFrame.pt);
		//drawOpticalFlow(show, lastFrame.pt, lbp_pt);
		drawOpticalFlow(show2, lastFrame.pt, currentFrame.pt);

		putText(show2, //"FPS: " + to_string((1 / ((loop_end - loop_start) / (double)(CLOCKS_PER_SEC)))) +
			"  R_err: " + to_string(r_errCurr) +
			"  t_err: " + to_string(t_errCurr) +
			"  Error_R: " + to_string(R_err) +
			"  Error_t: " + to_string(t_err), Point(5, lastFrame.img.rows - 40), 1, 1, Scalar(0, 255, 0), 2);
		imshow("test", show2);

		showTrajAndGTruth(traj, vo.T_messure, vo.T);

		if (isCorrect) {
			lastFrame = currentFrame;
		}

		waitKey(1);
	}
}

void 
testingAndCheck() 
{
	struct VisualOdometry vo;
	loadPoses(vo.poses_gt);
	trajectoryDistances(vo.poses_gt, vo.trajDistances);
	initializePose(firstFrameID, vo);
	Mat traj = Mat::zeros(ROW, COL, CV_8UC3);
	bool findKeyframe = false;
	int keyframeCounter = 0;


	int frameID_1[2] = { 551, 552 };
	int frameID_2[2] = { 65, 66 };
	int frameID_3[2] = { 65, 67 };
	int frameID_4[2] = { 00, 03 };
	int *testFrameID = frameID_4;
	clock_t t1_1, t1_2;


	struct Frame lastFrame;

	//--- new test
	//struct Frame currentFrame = odometryInitial(vo, lastFrame);
	struct Frame currentFrame = odometryInitial(vo, lastFrame, vo.status);

	vo.poses_messure.push_back(vo.T_messure.clone());

	//--- make projection matrix
	currentFrame.P = makeProjectionMatrix(vo.T_messure);
	
	vector<Point3d> pts3D;
	triangulate_Points(vo.P_Keyframe, currentFrame.P, vo.ptKF, currentFrame.pt, vo.status, pts3D);
	vo.points3D_Vec.push_back(pts3D);

	vector<Point2f> pts1_ = lastFrame.pt, pts2_ = currentFrame.pt;
	//---------------------------------------------------------------------------------------------

	frameNum = 0;
	struct VisualOdometry vo2;
	loadPoses(vo2.poses_gt);
	trajectoryDistances(vo2.poses_gt, vo2.trajDistances);
	initializePose(firstFrameID, vo2);

	struct Frame lastFrame2;
	struct Frame currentFrame2 = odometryInitial(vo2, lastFrame2);

	vo2.poses_messure.push_back(vo2.T_messure.clone());
	int kfID2 = lastFrame2.id;

	//--- make projection matrix
	currentFrame2.P = makeProjectionMatrix(vo2.T_messure);

	Mat points4D;
	triangulatePoints(vo2.P_Keyframe, currentFrame2.P, vo2.ptKF, currentFrame2.pt, points4D);

	//--- normalize 4D points to 3D points and save to pt3DVec...
	vector<Point3d> points3D;
	normalizePoints(points4D, points3D);
	vo2.points3D_Vec.push_back(points3D);

	//--- Chierality check
	vector<uchar> status;
	for (unsigned int i = 0; i < points3D.size(); i++)
		status.push_back((points3D[i].z > 0) ? 1 : 0);

	cout << "triangulation likes " << countNonZero(Mat(status))
		 << " out of " << currentFrame2.pt.size()
		 << " (" << (float)(countNonZero(Mat(status))) / (float)(currentFrame2.pt.size()) * 100 << "%)" << endl << endl;

	// remove bad points.
	remove_bad_points(lastFrame2.pt, status);
	remove_bad_points(currentFrame2.pt, status);
	remove_bad_points(vo2.ptKF, status);
	remove_bad_points(vo2.points3D_Vec, status);


	remove_bad_points(pts1_, vo.status);
	remove_bad_points(pts2_, vo.status);
	remove_bad_points(vo.points3D_Vec, vo.status);


	printf("stop");

	/*getFrame(lastFrame, testFrameID[0]);
	detectKeypoints(lastFrame.grayImg, lastFrame.pt);

	int keyFrameSize = lastFrame.pt.size();
	vo.ptSizeKF_TH = lastFrame.pt.size();
	vo.ptKF = lastFrame.pt;
	int kfID = lastFrame.id;

	bool have_3DPt = false;


	struct Frame currentFrame;
	getFrame(currentFrame, testFrameID[1]);


	bool isKeyFrame = trackKeypoints(lastFrame.grayImg, currentFrame.grayImg,
		lastFrame.pt, currentFrame.pt,
		vo.ptKF, vo.ptSizeKF_TH, vo.points3D_Vec, have_3DPt);


	Mat T_f;
	bool isCorrect = poseEstimate(lastFrame.pt, currentFrame.pt, vo.ptKF, vo.points3D_Vec, T_f, have_3DPt);

	if (isCorrect) {
		have_3DPt = true;

		double scale = vo.trajDistances[currentFrame.id] - vo.trajDistances[kfID];
		Mat T = vo.T_messure.clone();
		updataPose(scale, T_f, T);

		vector<errors> err_f;
		Mat T_f_temp = T_f.clone();
		Mat t = T_f_temp(Range(0, 3), Range(3, 4));
		t *= scale;
		poseFrameErrors(vo.poses_gt, T_f_temp, kfID, currentFrame.id, err_f);
		printf("R: %f   t: %f \n", err_f[0].r_err, err_f[0].t_err);
	}*/
}