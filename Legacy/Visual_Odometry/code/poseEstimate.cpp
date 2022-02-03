#include "poseEstimate.h"

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
remove_bad_points(vector<Point3d> &pts, const vector<uchar> &status)
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

static inline void 
Rt2T(const Mat &_R, const Mat &_t, Mat &_T) 
{
	_T = Mat::eye(4, 4, CV_64F);
	_R.copyTo(_T(Range(0, 3), Range(0, 3)));
	_t.copyTo(_T(Range(0, 3), Range(3, 4)));
}

static int 
discardOutliersKeyPoints(vector<Point2f> &pt1, vector<Point2f> &pt2, const Mat &mask, vector<Point2f> &pt1_temp, vector<Point2f> &pt2_temp)
{
	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<unsigned char>(i)) {
			pt1_temp.push_back(pt1.at(i));
			pt2_temp.push_back(pt2.at(i));
		}
	}

	return pt2.size();
}

static int
discardOutliersKeyPoints(vector<Point2f> &pt1, vector<Point2f> &pt2, const Mat &mask)
{
	vector<Point2f> pt1_temp, pt2_temp;

	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<unsigned char>(i)) {
			pt1_temp.push_back(pt1.at(i));
			pt2_temp.push_back(pt2.at(i));
		}
	}

	pt1 = pt1_temp;
	pt2 = pt2_temp;

	return pt2.size();
}

static int
discardOutliersKeyPoints(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &pt3, const Mat &mask)
{
	vector<Point2f> pt1_temp, pt2_temp, pt3_temp;

	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<unsigned char>(i)) {
			pt1_temp.push_back(pt1.at(i));
			pt2_temp.push_back(pt2.at(i));
			pt3_temp.push_back(pt3.at(i));
		}
	}

	pt1 = pt1_temp;
	pt2 = pt2_temp;
	pt3 = pt3_temp;

	return pt2.size();
}

static int
discardOutliersKeyPoints(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &pt3, vector< vector<Point3d> > &pt3D_Vec, const Mat &mask)
{
	vector<Point2f> pt1_temp, pt2_temp, pt3_temp;
	const int size = pt3D_Vec.size();
	vector< vector<Point3d> > pt3D_Vec_temp(size);

	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<unsigned char>(i)) {
			pt1_temp.push_back(pt1.at(i));
			pt2_temp.push_back(pt2.at(i));
			pt3_temp.push_back(pt3.at(i));

			for (int j = 0; j < size; j++) {
				pt3D_Vec_temp[j].push_back(pt3D_Vec[j].at(i));
			}
		}
	}

	pt1 = pt1_temp;
	pt2 = pt2_temp;
	pt3 = pt3_temp;
	pt3D_Vec = pt3D_Vec_temp;

	return pt2.size();
}

static inline int 
inliersKPSize(const Mat &_mask) 
{
	int counter = 0;
	for (int i = 0; i < _mask.rows; i++) {
		if (_mask.at<unsigned char>(i)) {
			counter++;
		}
	}

	return counter;
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

static float 
CheckFundamental(const Mat &_F, const vector<Point2f> &_pt1, const vector<Point2f> &_pt2, Mat &_mask_F)
{
	CV_Assert( _pt1.size() == _pt2.size() );
	//const int N = _pt1.size();
	const int N = _mask_F.rows;

	const float F11 = _F.at<double>(0, 0);
	const float F12 = _F.at<double>(0, 1);
	const float F13 = _F.at<double>(0, 2);
	const float F21 = _F.at<double>(1, 0);
	const float F22 = _F.at<double>(1, 1);
	const float F23 = _F.at<double>(1, 2);
	const float F31 = _F.at<double>(2, 0);
	const float F32 = _F.at<double>(2, 1);
	const float F33 = _F.at<double>(2, 2);

	//vbMatchesInliers.resize(N);
	_mask_F.create(N, 1, CV_8UC1);
	_mask_F.setTo(Scalar::all(1));
	uchar *maskPtr = _mask_F.ptr<uchar>();

	float score = 0;

	const float th = /*3.841*/maxDistanceErr_F;		// 95% => 3.84; 99% => 6.63;
	const float thScore = 5.991;

	const float invSigmaSquare = 1.0 / (sigma*sigma);

	for (int i = 0; i<N; i++)
	{
		//if ( *(maskPtr + i) == 0 ) {
			//continue;
		//}
		bool bIn = true;

		const Point2f point1 = _pt1[i];
		const Point2f point2 = _pt2[i];

		const float u1 = point1.x;
		const float v1 = point1.y;
		const float u2 = point2.x;
		const float v2 = point2.y;

		// Reprojection error in second image
		// l2 = F21x1 = (a2,b2,c2)

		const float a2 = F11*u1 + F12*v1 + F13;
		const float b2 = F21*u1 + F22*v1 + F23;
		const float c2 = F31*u1 + F32*v1 + F33;

		const float num2 = a2*u2 + b2*v2 + c2;

		const float squareDist1 = num2*num2 / (a2*a2 + b2*b2);

		const float chiSquare1 = squareDist1*invSigmaSquare;

		if (chiSquare1 > th)
			bIn = false;
		else
			//score += thScore - chiSquare1;
			//score += chiSquare1;
			;

		// Reprojection error in second image
		// l1 = x2tF21 = (a1,b1,c1)

		const float a1 = F11*u2 + F21*v2 + F31;
		const float b1 = F12*u2 + F22*v2 + F32;
		const float c1 = F13*u2 + F23*v2 + F33;

		const float num1 = a1*u1 + b1*v1 + c1;

		const float squareDist2 = num1*num1 / (a1*a1 + b1*b1);

		const float chiSquare2 = squareDist2*invSigmaSquare;

		if (chiSquare2 > th)
			bIn = false;
		else
			//score += thScore - chiSquare2;
			//score += chiSquare2;
			;

		if (bIn)
			score++;

		if (!bIn)
			*(maskPtr + i) = 0;
	}
	score /= N;

	return score;
}

static float 
CheckHomography(const cv::Mat &_H21, const cv::Mat &_H12, const vector<Point2f> &_pt1, const vector<Point2f> &_pt2)
{
	CV_Assert(_pt1.size() == _pt2.size());
	const int N = _pt1.size();

	const float h11 = _H21.at<double>(0, 0);
	const float h12 = _H21.at<double>(0, 1);
	const float h13 = _H21.at<double>(0, 2);
	const float h21 = _H21.at<double>(1, 0);
	const float h22 = _H21.at<double>(1, 1);
	const float h23 = _H21.at<double>(1, 2);
	const float h31 = _H21.at<double>(2, 0);
	const float h32 = _H21.at<double>(2, 1);
	const float h33 = _H21.at<double>(2, 2);

	const float h11inv = _H12.at<double>(0, 0);
	const float h12inv = _H12.at<double>(0, 1);
	const float h13inv = _H12.at<double>(0, 2);
	const float h21inv = _H12.at<double>(1, 0);
	const float h22inv = _H12.at<double>(1, 1);
	const float h23inv = _H12.at<double>(1, 2);
	const float h31inv = _H12.at<double>(2, 0);
	const float h32inv = _H12.at<double>(2, 1);
	const float h33inv = _H12.at<double>(2, 2);

	//vbMatchesInliers.resize(N);

	float score = 0;

	const float th = 5.991;

	const float invSigmaSquare = 1.0 / (sigma*sigma);

	for (int i = 0; i<N; i++)
	{
		bool bIn = true;

		const Point2f point1 = _pt1[i];
		const Point2f point2 = _pt2[i];

		const float u1 = point1.x;
		const float v1 = point1.y;
		const float u2 = point2.x;
		const float v2 = point2.y;

		// Reprojection error in first image
		// x2in1 = H12*x2

		const float w2in1inv = 1.0 / (h31inv*u2 + h32inv*v2 + h33inv);
		const float u2in1 = (h11inv*u2 + h12inv*v2 + h13inv)*w2in1inv;
		const float v2in1 = (h21inv*u2 + h22inv*v2 + h23inv)*w2in1inv;

		const float squareDist1 = (u1 - u2in1)*(u1 - u2in1) + (v1 - v2in1)*(v1 - v2in1);

		const float chiSquare1 = squareDist1*invSigmaSquare;

		if (chiSquare1>th)
			bIn = false;
		else
			score += th - chiSquare1;

		// Reprojection error in second image
		// x1in2 = H21*x1

		const float w1in2inv = 1.0 / (h31*u1 + h32*v1 + h33);
		const float u1in2 = (h11*u1 + h12*v1 + h13)*w1in2inv;
		const float v1in2 = (h21*u1 + h22*v1 + h23)*w1in2inv;

		const float squareDist2 = (u2 - u1in2)*(u2 - u1in2) + (v2 - v1in2)*(v2 - v1in2);

		const float chiSquare2 = squareDist2*invSigmaSquare;

		if (chiSquare2>th)
			bIn = false;
		else
			score += th - chiSquare2;

		//if (bIn)
			//vbMatchesInliers[i] = true;
		//else
			//vbMatchesInliers[i] = false;
	}

	return score;
}

static Mat
computeF(const vector<Point2f> &_pt1, const vector<Point2f> &_pt2, Mat &_mask)
{
	CV_Assert(_pt1.size() == _pt2.size());
	const int N = _pt1.size();
	int runTimes = 0;
	vector<Point2f> pt1, pt2;
	pt1 = _pt1;
	pt2 = _pt2;

	srand((unsigned)time(NULL));
	
	Mat F_final = findFundamentalMat(pt2, pt1, FM_RANSAC, maxDistanceErr_F, 0.995, _mask);
	int test1 = countNonZero(_mask);
	float inliersRatio = CheckFundamental(F_final, _pt2, _pt1, _mask);
	int test2 = countNonZero(_mask);
	float maxRatio = inliersRatio;

	while (/*inliersRatio < 0.9 &&*/ runTimes < 0) {
		runTimes++;
		pt1.clear();
		pt2.clear();

		for (int i = 0; i < N; i++) {
			int idx = rand() % N;
			pt1.push_back(_pt1[idx]);
			pt2.push_back(_pt2[idx]);
		}

		Mat mask;
		Mat F = findFundamentalMat(pt2, pt1, FM_RANSAC, maxDistanceErr_F, 0.995, mask);
		inliersRatio = CheckFundamental(F, _pt2, _pt1, mask);
		//printf("inliersRatio(F): %f \n", inliersRatio);

		if (inliersRatio > maxRatio) {
			F_final = F.clone();
			_mask = mask.clone();
			maxRatio = inliersRatio;
		}

		/*if (inliersRatio < maxRatio) {
			F_final = F.clone();
			_mask = mask.clone();
			maxRatio = inliersRatio;
		}*/
	}
	//printf("inliersRatio(F): %f \n", maxRatio);

	return F_final;
}

static vector<uchar> 
rebuildStatus(vector<uchar> &originStatus, vector<uchar> &newStatus)
{
	vector<uchar> status;
	const int N = originStatus.size();
	size_t i, k = 0;

	for (i = k = 0; i < N; i++) {
		if (!originStatus[i]) {
			status.push_back(0);
			continue;
		}
		status.push_back(newStatus[k]);
		k = k + 1;
	}

	return status;
}

static float
CheckFundamental(const Mat &F, const vector<Point2f> &pts1, const vector<Point2f> &pts2, vector<uchar> &status)
{
	CV_Assert(pts1.size() == pts2.size());
	const int N = status.size();

	const float F11 = F.at<double>(0, 0);
	const float F12 = F.at<double>(0, 1);
	const float F13 = F.at<double>(0, 2);
	const float F21 = F.at<double>(1, 0);
	const float F22 = F.at<double>(1, 1);
	const float F23 = F.at<double>(1, 2);
	const float F31 = F.at<double>(2, 0);
	const float F32 = F.at<double>(2, 1);
	const float F33 = F.at<double>(2, 2);

	float score = 0;
	float error = 0;

	const float th = /*3.841*/maxDistanceErr_F;		// 95% => 3.84; 99% => 6.63;
	const float thScore = 5.991;

	const float invSigmaSquare = 1.0 / (sigma * sigma);

	for (int i = 0; i<N; i++)
	{
		if (status[i] == 0) {
			continue;
		}
		else {
			bool bIn = true;

			const Point2f point1 = pts1[i];
			const Point2f point2 = pts2[i];

			const float u1 = point1.x;
			const float v1 = point1.y;
			const float u2 = point2.x;
			const float v2 = point2.y;

			// Reprojection error in second image
			// l2 = F21x1 = (a2,b2,c2)

			const float a2 = F11*u1 + F12*v1 + F13;
			const float b2 = F21*u1 + F22*v1 + F23;
			const float c2 = F31*u1 + F32*v1 + F33;

			const float num2 = a2*u2 + b2*v2 + c2;

			const float squareDist1 = num2*num2 / (a2*a2 + b2*b2);

			const float chiSquare1 = squareDist1*invSigmaSquare;

			if (chiSquare1 > th)
				bIn = false;
			else
				//score += thScore - chiSquare1;
				error += chiSquare1;
				//;

			// Reprojection error in second image
			// l1 = x2tF21 = (a1,b1,c1)

			const float a1 = F11*u2 + F21*v2 + F31;
			const float b1 = F12*u2 + F22*v2 + F32;
			const float c1 = F13*u2 + F23*v2 + F33;

			const float num1 = a1*u1 + b1*v1 + c1;

			const float squareDist2 = num1*num1 / (a1*a1 + b1*b1);

			const float chiSquare2 = squareDist2*invSigmaSquare;

			if (chiSquare2 > th)
				bIn = false;
			else
				//score += thScore - chiSquare2;
				error += chiSquare2;
				//;

			//if (bIn)
				//score++;

			if (!bIn)
				status[i] = 0;
		}
	}
	//score /= N;

	//return score;
	return error;
}

static Mat
computeF(const vector<Point2f> &pts1, const vector<Point2f> &pts2, vector<uchar> &status)
{
	CV_Assert(pts1.size() == pts2.size());

	vector<Point2f> pts1_temp = pts1, pts2_temp = pts2;
	remove_bad_points(pts1_temp, status);
	remove_bad_points(pts2_temp, status);

	vector<uchar> status_F;
	Mat F = findFundamentalMat(pts2_temp, pts1_temp, FM_RANSAC, maxDistanceErr_F, 0.995, status_F);
	float error = CheckFundamental(F, pts2_temp, pts1_temp, status_F);
	//printf("fundamentalMat error = %f \n\n", error);
	status = rebuildStatus(status, status_F);
	CV_Assert(pts1.size() == status.size());

	return F;
}

void 
poseInitial(Mat &R, Mat &t) 
{
	R = Mat::eye(Size(3, 3), CV_64FC1);
	t = Mat::zeros(Size(1, 3), CV_64FC1);
}

bool 
poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, Mat &R_f, Mat &t_f, const bool keyFrame)
{
	Mat E, mask;
	//vector<Point2f> pt1_temp, pt2_temp;

	E = findEssentialMat(pt2, pt1, focal, pp, RANSAC, 0.999, 1.0, mask);
	//int numKeyPoints = discardOutliersKeyPoints(pt1, pt2, mask, pt1_temp, pt2_temp);
	int numKeyPoints = discardOutliersKeyPoints(pt1, pt2, mask);
	mask.release();
	if (!keyFrame) {
		return false;
	}

	int numInliers = recoverPose(E, pt2, pt1, R_f, t_f, focal, pp, mask);
	
	//printf("Inliers: %d(%d) \n \n", numInliers, numKeyPoints);
	printf("Inliers Ratio: %f \n", 100 * (float)(numInliers)/numKeyPoints);
	//cout << R << endl << endl << t << endl << endl;

	if ( (numInliers > (numKeyPoints * poseCorrectRatio)) && (t_f.at<double>(2) > 2 * t_f.at<double>(0)) && (t_f.at<double>(2) > 2 * t_f.at<double>(1))) {
		//pt1 = pt1_temp;
		//pt2 = pt2_temp;

		return true;
	}
	else {
		return false;
	}
}

bool
poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, Mat &T, const bool keyFrame)
{
	Mat E, mask;
	Mat R_f, t_f;
	//vector<Point2f> pt1_temp, pt2_temp;

	E = findEssentialMat(pt2, pt1, focal, pp, RANSAC, 0.999, 1.0, mask);
	//int numKeyPoints = discardOutliersKeyPoints(pt1, pt2, mask, pt1_temp, pt2_temp);
	int numKeyPoints = discardOutliersKeyPoints(pt1, pt2, mask);
	mask.release();
	if (!keyFrame) {
		return false;
	}

	int numInliers = recoverPose(E, pt2, pt1, R_f, t_f, focal, pp, mask);
	T = Mat::eye(4, 4, CV_64F);
	R_f.copyTo(T(Range(0, 3), Range(0, 3)));
	t_f.copyTo(T(Range(0, 3), Range(3, 4)));

	//printf("Inliers: %d(%d) \n \n", numInliers, numKeyPoints);
	printf("Inliers Ratio: %f \n", 100 * (float)(numInliers) / numKeyPoints);
	//cout << R << endl << endl << t << endl << endl;

	if ((numInliers > (numKeyPoints * poseCorrectRatio)) && (t_f.at<double>(2) > 2 * t_f.at<double>(0)) && (t_f.at<double>(2) > 2 * t_f.at<double>(1))) {
		//pt1 = pt1_temp;
		//pt2 = pt2_temp;

		return true;
	}
	else {
		return false;
	}
}

bool
poseEstimate(vector<Point2f> &pts1, vector<Point2f> &pts2, vector<uchar> &status, Mat &T, const bool keyFrame)
{	
	Mat F = computeF(pts1, pts2, status);
	int numKeyPoints = countNonZero(status);
	if (!keyFrame) {
		return false;
	}

	Mat E = K.t() * F * K;

	Mat  R_f, t_f;
	vector<uchar> status4Pose/* = status*/;

	int numInliers = recoverPose(E, pts2, pts1, R_f, t_f, focal, pp, status4Pose);
	Rt2T(R_f, t_f, T);

	double normT = normOfTransform(R_f, t_f);

	if ((numInliers > (numKeyPoints * poseCorrectRatio)) && (normT < 2.5) &&
		(t_f.at<double>(2) > 2 * t_f.at<double>(0)) && (t_f.at<double>(2) > 2 * t_f.at<double>(1))) {

		//printf("Inliers Ratio: %f \n", 100 * (float)(numInliers) / numKeyPoints);
		return true;
	}
	else {
		return false;
	}
}

bool
poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, Mat &T, const bool keyFrame)
{
	//--- for testing
	//if (!keyFrame) {
		//return false;
	//}

	Mat mask_F;
	Mat F = computeF(ptKF, pt2, mask_F);
	int numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, mask_F);

	if (!keyFrame) {
		return false;
	}

	Mat E = K.t() * F * K;

	Mat  R_f, t_f, mask;
	int numInliers = recoverPose(E, pt2, ptKF, R_f, t_f, focal, pp, mask);
	Rt2T(R_f, t_f, T);

	double normT = normOfTransform(R_f, t_f);
	//printf("norm T: %lf \n\n", normT);
	
	/*Mat E, mask;
	Mat R_f, t_f;
	vector<Point2f> pt1_temp, pt2_temp;

	E = findEssentialMat(pt2, ptKF, focal, pp, RANSAC, 0.999, 1.0, mask);
	int numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, mask);*/
	//int numKeyPoints = discardOutliersKeyPoints(pt1, pt2, mask, pt1_temp, pt2_temp);

	/*Mat F = findFundamentalMat(pt2, ptKF, FM_RANSAC, 1.0, 0.995);
	Mat mask_F;
	const float scoreF = CheckFundamental(F, pt2, ptKF, mask_F);
	int numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, mask_F);*/
	/*mask.release();
	if (!keyFrame) {
		return false;
	}*/

	//const Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
	//E = K.t() * F * K;

	//int numInliers = recoverPose(E, pt2, ptKF, R_f, t_f, focal, pp, mask);
	//Rt2T(R_f, t_f, T);
	//Mat r_vec;
	//Rodrigues(R_f, r_vec);
	//cout << "R(Norm): " << norm(r_vec) << endl;

	//printf("Inliers: %d(%d) \n \n", numInliers, numKeyPoints);
	//printf("Inliers Ratio: %f \n", 100 * (float)(numInliers) / numKeyPoints);
	//cout << R << endl << endl << t << endl << endl;
	
	if ((numInliers > (numKeyPoints * poseCorrectRatio)) && (normT < 2.5) &&
		(t_f.at<double>(2) > 2 * t_f.at<double>(0)) && (t_f.at<double>(2) > 2 * t_f.at<double>(1))) {
		//pt1 = pt1_temp;
		//pt2 = pt2_temp;
		printf("Inliers Ratio: %f \n", 100 * (float)(numInliers) / numKeyPoints);
		return true;
	}
	else {
		return false;
	}
}

bool
poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, Mat &T)
{
	/*Mat F = findFundamentalMat(pt2, ptKF, FM_RANSAC, 1.0, 0.995);
	
	Mat mask_F;
	const float scoreF = CheckFundamental(F, pt2, ptKF, mask_F);
	int numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, mask_F);

	Mat E = K.t() * F * K;*/

	Mat mask_E;
	Mat E = findEssentialMat(pt2, ptKF, focal, pp, RANSAC, 0.999, 1.0, mask_E);
	int numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, mask_E);

	Mat  R_f, t_f, mask;
	int numInliers = recoverPose(E, pt2, ptKF, R_f, t_f, focal, pp, mask);
	Rt2T(R_f, t_f, T);

	if ((numInliers > (numKeyPoints * poseCorrectRatio)) &&
		(t_f.at<double>(2) > 2 * t_f.at<double>(0)) && (t_f.at<double>(2) > 2 * t_f.at<double>(1))) {
		//pt1 = pt1_temp;
		//pt2 = pt2_temp;
		printf("Inliers Ratio: %f \n", 100 * (float)(numInliers) / numKeyPoints);
		return true;
	}
	else {
		return false;
	}
}

bool
poseEstimate(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, vector< vector<Point3d> > &pt3D_Vec, Mat &T, const bool &have_3DPt)
{
	int kpSize = ptKF.size();
	Mat mask_F;
	Mat F = computeF(ptKF, pt2, mask_F);
	//Mat mask_F_test;
	//Mat F_test = findFundamentalMat(pt2, ptKF, FM_RANSAC, 1.0, 0.995, mask_F_test);
	//Mat F = findFundamentalMat(pt2, ptKF, FM_RANSAC, 0.2, 0.995, mask_F);

	/*float scoreF = CheckFundamental(F, pt2, ptKF, mask_F);
	cout << scoreF << endl;
	int numKeyPoints;
	while (scoreF < 0.95) {
		if (!have_3DPt) {
			numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, mask_F);
		}
		else {
			numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, pt3D_Vec, mask_F);
		}

		mask_F.release();
		F = findFundamentalMat(pt2, ptKF, FM_RANSAC, 1.0, 0.995, mask_F);
		scoreF = CheckFundamental(F, pt2, ptKF, mask_F);
		cout << scoreF << endl;
	}
	cout << endl;*/
	int numKeyPoints;
	//int numKeyPoints = countNonZero(mask_F);
	if (!have_3DPt) {
		numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, mask_F);
	}
	else {
		numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, pt3D_Vec, mask_F);
	}
	//int numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, mask_F);

	Mat E = K.t() * F * K;

	/*Mat mask_E;
	Mat E = findEssentialMat(pt2, ptKF, focal, pp, RANSAC, 0.999, 1.0, mask_E);
	
	int numKeyPoints;
	if (!have_3DPt) {
		numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, mask_E);
	}
	else {
		numKeyPoints = discardOutliersKeyPoints(ptKF, pt2, pt1, pt3D_Vec, mask_E);
	}*/

	//int numKeyPoints = inliersKPSize(mask_E);
	//int numKeyPoints = inliersKPSize(mask_F);
	
	Mat  R_f, t_f, mask;
	int numInliers = recoverPose(E, pt2, ptKF, R_f, t_f, focal, pp, mask);
	//int numInliers = recoverPose(E, pt2, ptKF, R_f, t_f, focal, pp, mask_E);
	//int numInliers = recoverPose(E, pt2, ptKF, R_f, t_f, focal, pp, mask_F);
	Rt2T(R_f, t_f, T);

	double normT = normOfTransform(R_f, t_f);
	//printf("norm T: %lf \n\n", normT);

	if ((numInliers > (numKeyPoints * poseCorrectRatio)) && (normT < 2.5) &&
		(t_f.at<double>(2) > 2.0 * t_f.at<double>(0)) && (t_f.at<double>(2) > 2.0 * t_f.at<double>(1))) {
		//pt1 = pt1_temp;
		//pt2 = pt2_temp;
		//printf("Inliers Ratio: %f \n", 100 * (float)(numInliers) / numKeyPoints);
		return true;
	}
	else {
		return false;
	}
}

bool 
poseEstimate(vector<Point2f> &_pt1, vector<Point2f> &_pt2, Mat &_T)
{
	Mat R_f, t_f, mask, mask_F, mask_H;
	vector<Mat> R_fVec, t_fVec, normals;
	
	Mat F = findFundamentalMat(_pt2, _pt1, FM_RANSAC, 1.0, 0.995);
	const float scoreF = CheckFundamental(F, _pt2, _pt1, mask_F);
	

	Mat H = findHomography(_pt2, _pt1, RANSAC, 1.0, mask_H, 2000, 0.995);
	const float scoreH = CheckHomography(H, H.inv(), _pt2, _pt1);

	
	const float RH = scoreH / (scoreH + scoreF);
	printf("RH: %f  SF: %f  SH: %f \n", RH, scoreF, scoreH);

	
	const Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
	// Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45), else if => (pF_HF>0.6)
	if (RH > 0.40) {
		int n = decomposeHomographyMat(H, K, R_fVec, t_fVec, Mat());
		Rt2T(R_f, t_f, _T);
		return true;
	}
	else {
		//Mat E_ = findEssentialMat(_pt2, _pt1, focal, pp, RANSAC, 0.999, 1.0, mask);
		//Mat R_f_, t_f_;
		//int numInliers_ = recoverPose(E_, _pt2, _pt1, R_f_, t_f_, focal, pp, mask);
		//mask.release();

		//const int ptSize = discardOutliersKeyPoints(_pt1, _pt2, mask_F);
		Mat E = K.t() * F * K;
		int numInliers = recoverPose(E, _pt2, _pt1, R_f, t_f, focal, pp, mask);
		//printf("numInliers: %d(%f) \n\n", numInliers, (float)numInliers/_pt2.size());

		if ( /*(numInliers > (ptSize * poseCorrectRatio)) && */
			 (t_f.at<double>(2) > 2 * t_f.at<double>(0)) && (t_f.at<double>(2) > 2 * t_f.at<double>(1)) ) {
			printf("numInliers: %d(%f) \n\n", numInliers, (float)numInliers / _pt2.size());
			Rt2T(R_f, t_f, _T);
			return true;
		}
		else {
			int n = decomposeHomographyMat(H, K, R_fVec, t_fVec, normals);
			printf("Use Homography Matrix \n\n");
			Rt2T(R_fVec[0], t_fVec[0], _T);
			return true;
		}
	}

	return false;
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

Mat 
poseEstimatePnP(const vector<Point3d> &pts3D, const vector<Point2f> &pts, const vector<uchar> &status) 
{
	vector<Point3d> pts3D_ = pts3D;
	vector<Point2f> pts_ = pts;

	// remove bad points.
	remove_bad_points(pts3D_, status);
	remove_bad_points(pts_, status);
	
	//--- solve PnP for estimating camera pose.
	Mat r_pnp, t_pnp, R_pnp;
	//solvePnP(vo.points3D_Vec[vo.points3D_Vec.size()-1], nextFrame.pt, K, Mat(), r_pnp, t_pnp, false, SOLVEPNP_ITERATIVE);
	solvePnPRansac(pts3D_, pts_, K, Mat(), r_pnp, t_pnp, false, 5000, 1.0, 0.99);

	//--- inverse transformation from solvePnP.
	Mat T_pnp = inverseTransformation(r_pnp, t_pnp);

	return T_pnp;
}