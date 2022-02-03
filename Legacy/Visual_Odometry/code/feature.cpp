#include "feature.h"


static inline bool 
response_comparator(const KeyPoint &p1, const KeyPoint &p2) 
{
	return p1.response > p2.response;
}

static void 
HarrisResponses(const Mat &img, vector<KeyPoint> &pts, int blockSize, const float &harris_k)
{
	CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);

	size_t ptidx, ptsize = pts.size();

	const uchar *ptr00 = img.ptr<uchar>();
	int step = (int)(img.step / img.elemSize1());
	int r = blockSize / 2;

	float scale = 1.f / ((1 << 2) * blockSize * 255.f);
	float scale_sq_sq = scale * scale * scale * scale;

	AutoBuffer<int> ofsbuf(blockSize*blockSize);
	int *ofs = ofsbuf;
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			ofs[i*blockSize + j] = (int)(i*step + j);

	for (ptidx = 0; ptidx < ptsize; ptidx++) {
		int x0 = cvRound(pts[ptidx].pt.x);
		int y0 = cvRound(pts[ptidx].pt.y);
		//int z = pts[ptidx].octave;

		//const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
		const uchar *ptr0 = ptr00 + (y0 - r)*step + x0 - r;
		int a = 0, b = 0, c = 0;

		for (int k = 0; k < blockSize*blockSize; k++) {
			const uchar *ptr = ptr0 + ofs[k];
			int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);
			int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);
			a += Ix*Ix;
			b += Iy*Iy;
			c += Ix*Iy;
		}
		pts[ptidx].response = ((float)a * b - (float)c * c -
			harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
	}
}

static void
conerResponses(const Mat &img, vector<KeyPoint> &pts, int blockSize)
{
	CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);

	size_t ptidx, ptsize = pts.size();

	const uchar *ptr00 = img.ptr<uchar>();
	int step = (int)(img.step / img.elemSize1());
	int r = blockSize / 2;

	AutoBuffer<int> ofsbuf(blockSize*blockSize);
	int *ofs = ofsbuf;
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			ofs[i*blockSize + j] = (int)(i*step + j);

	for (ptidx = 0; ptidx < ptsize; ptidx++) {
		int x0 = cvRound(pts[ptidx].pt.x);
		int y0 = cvRound(pts[ptidx].pt.y);
		//int z = pts[ptidx].octave;

		//const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
		const uchar *ptr0 = ptr00 + (y0 - r)*step + x0 - r;
		int positive = 0, negitive = 0;

		for (int k = 0; k < blockSize*blockSize; k++) {
			const uchar *ptr = ptr0 + ofs[k];
			positive += (ptr[-step + 1] + ptr[-step + 2] + (ptr[-2 * step + 1] + ptr[-2 * step + 2])) + (ptr[step - 1] + ptr[step - 2] + (ptr[2 * step - 1] + ptr[2 * step - 2]));
			negitive += (ptr[-step - 1] + ptr[-step - 2] + (ptr[-2 * step - 1] + ptr[-2 * step - 2])) + (ptr[step + 1] + ptr[step + 2] + (ptr[2 * step + 1] + ptr[2 * step + 2]));
		}
		pts[ptidx].response = abs(positive + negitive);
	}
}

void
detectKeypoints(const Mat &img, vector<Point2f> &pt)
{
	const int minBorderX = edgeThreshold;
	const int minBorderY = minBorderX;
	const int maxBorderX = img.cols - edgeThreshold;
	const int maxBorderY = img.rows - edgeThreshold;

	const float width = (maxBorderX - minBorderX);
	const float height = (maxBorderY - minBorderY);

	const int nCols = width / 100;
	const int nRows = height / 50;
	const int wCell = ceil(width / nCols);
	const int hCell = ceil(height / nRows);

	const int nCells = nRows * nCols;
	//const int nfeaturesCell = ceil((float)MaxFeaturesNum / nCells);
	//const int nfeaturesCell = 13;	// [retainBest => x2] KITTI_07 => 5; other => 2  [retainBest => x1] KITTI_07 => 10(lost para), 13(best); other => 4;

	vector<KeyPoint> keypoints;
	for (int i = 0; i < nRows; i++) {
		const float iniY = minBorderY + i * hCell;
		float maxY = iniY + hCell + 6;

		if (iniY >= maxBorderY - 3)
			continue;
		if (maxY > maxBorderY)
			maxY = maxBorderY;

		for (int j = 0; j < nCols; j++) {
			const float iniX = minBorderX + j * wCell;
			float maxX = iniX + wCell + 6;

			if (iniX >= maxBorderX - 6)
				continue;
			if (maxX>maxBorderX)
				maxX = maxBorderX;

			vector<Point2f> vPtsCell;
			//goodFeaturesToTrack(img.rowRange(iniY, maxY).colRange(iniX, maxX), vPtsCell, 100, 0.01, 10, Mat(), 3, false, 0.04);
			goodFeaturesToTrack(img.rowRange(iniY, maxY).colRange(iniX, maxX), vPtsCell, maxCorners, qualityLevel, minDistance, Mat(), detectBlockSize, false, 0.04);

			vector<KeyPoint> vKeysCell;
			if (!vPtsCell.empty()) {
				KeyPoint::convert(vPtsCell, vKeysCell);
				HarrisResponses(img.rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, 7, HARRIS_K);
				//conerResponses(img.rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, 7);
				sort(vKeysCell.begin(), vKeysCell.end(), response_comparator);
				KeyPointsFilter::retainBest(vKeysCell, 1 * nfeaturesCell);

				for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++) {
					(*vit).pt.x += j*wCell;
					(*vit).pt.y += i*hCell;
					keypoints.push_back(*vit);
				}
			}

		}
	}

	// Add border to coordinates and scale information
	const int nkps = keypoints.size();
	for (int i = 0; i<nkps; i++) {
		keypoints[i].pt.x += minBorderX;
		keypoints[i].pt.y += minBorderY;
	}

	KeyPoint::convert(keypoints, pt);
}

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

/*static void 
KLT_Re_Check(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<uchar> &status)
{
	int indexCorrection = 0;

	for(int i = 0; i < status.size(); i++) {  
		Point2f pt = pt2.at(i- indexCorrection);
		if ( (status.at(i) == 0) || (pt.x<0)||(pt.y<0) ) {
			if((pt.x<0)||(pt.y<0)) {
				status.at(i) = 0;
			}
			pt1.erase (pt1.begin() + (i - indexCorrection));
			pt2.erase (pt2.begin() + (i - indexCorrection));
			indexCorrection++;
		}
	}
}*/

static void 
KLT_Re_Check(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<uchar> &status)
{
	vector<Point2f> pt1_temp, pt2_temp;

	for(int i = 0; i < status.size(); i++) {  
		Point2f pt = pt2.at(i);

		if ( (pt.x<0)||(pt.y<0) ) {
			status.at(i) = 0;
		} else {
			pt1_temp.push_back(pt1.at(i));
			pt2_temp.push_back(pt2.at(i));
		}
	}

	pt1 = pt1_temp;
	pt2 = pt2_temp;
}

static void
KLT_Re_Check(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, vector<uchar> &status)
{
	vector<Point2f> pt1_temp, pt2_temp, ptKF_temp;
	vector<float> distance;

	for (int i = 0; i < status.size(); i++) {
		Point2f pt = pt2.at(i);

		if ((pt.x<0) || (pt.y<0)) {
			status.at(i) = 0;
		}
		else {
			pt1_temp.push_back(pt1.at(i));
			pt2_temp.push_back(pt2.at(i));
			ptKF_temp.push_back(ptKF.at(i));
		}
	}

	pt1 = pt1_temp;
	pt2 = pt2_temp;
	ptKF = ptKF_temp;
}

static float
KLT_Re_Check(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, vector<uchar> &status, const int &width, const int &height)
{
	vector<Point2f> pt1_temp, pt2_temp, ptKF_temp;
	vector<float> distance;

	for (int i = 0; i < status.size(); i++) {
		Point2f pt = pt2.at(i);

		if ((pt.x<edgeThreshold) || (pt.y<edgeThreshold) || (pt.x>width - edgeThreshold) || (pt.y>height - edgeThreshold)) {
			status.at(i) = 0;
		}
		else {
			pt1_temp.push_back(pt1.at(i));
			pt2_temp.push_back(pt2.at(i));
			ptKF_temp.push_back(ptKF.at(i));

			distance.push_back(sqrt(powf(pt1.at(i).x - pt2.at(i).x, 2) + powf(pt1.at(i).y - pt2.at(i).y, 2)));
		}
	}

	pt1 = pt1_temp;
	pt2 = pt2_temp;
	ptKF = ptKF_temp;

	sort(distance.begin(), distance.end());
	float disMedian = distance.at(distance.size() / 2);
	return disMedian;
}

static float
KLT_Re_Check(vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, vector< vector<Point3d> > &pt3D_Vec, vector<uchar> &status, const int &width, const int &height)
{
	vector<Point2f> pt1_temp, pt2_temp, ptKF_temp;
	const int size = pt3D_Vec.size();
	vector< vector<Point3d> > pt3D_Vec_temp(size);
	vector<float> distance;

	for (int i = 0; i < status.size(); i++) {
		Point2f pt = pt2.at(i);

		if ((pt.x<edgeThreshold) || (pt.y<edgeThreshold) || (pt.x>width - edgeThreshold) || (pt.y>height - edgeThreshold)) {
			status.at(i) = 0;
		}
		else {
			pt1_temp.push_back(pt1.at(i));
			pt2_temp.push_back(pt2.at(i));
			ptKF_temp.push_back(ptKF.at(i));
				
			for (int j = 0; j < size; j++) {
				pt3D_Vec_temp[j].push_back(pt3D_Vec[j].at(i));
			}

			distance.push_back( sqrt( powf(pt1.at(i).x - pt2.at(i).x, 2) + powf(pt1.at(i).y - pt2.at(i).y, 2) ) );
		}
	}

	pt1 = pt1_temp;
	pt2 = pt2_temp;
	ptKF = ptKF_temp;
	pt3D_Vec = pt3D_Vec_temp;

	sort(distance.begin(), distance.end());
	float disMedian = distance.at(distance.size() / 2);
	return disMedian;
}

/*static void
Pts_Re_Check(const Mat &_img1, const Mat &_img2, 
			 vector<Point2f> &_pt1, vector<Point2f> &_pt2, vector<Point2f> &_ptKF, vector< vector<Point3d> > &_pt3D_Vec, const bool &_have3DPts, const float disMedian)
{
	CV_Assert( !(_pt1.size() == _pt2.size() == _ptKF.size()) );
	vector<Point2f> pt1_temp, pt2_temp, ptKF_temp, pt1_result, pt2_result, ptKF_result;
	const int N = _ptKF.size();
	const int size = _pt3D_Vec.size();
	vector< vector<Point3d> > pt3D_Vec_temp(size), pt3D_Vec_result(size);

	for (int i = 0; i < N; i++) {
		float distance = sqrt(powf(_pt1.at(i).x - _pt2.at(i).x, 2) + powf(_pt1.at(i).y - _pt2.at(i).y, 2));
		if (distance < 5 * disMedian) {
			pt1_temp.push_back(_pt1.at(i));
			pt2_temp.push_back(_pt2.at(i));
			ptKF_temp.push_back(_ptKF.at(i));

			if (_have3DPts) {
				for (int j = 0; j < size; j++) {
					pt3D_Vec_temp[j].push_back(_pt3D_Vec[j].at(i));
				}
			}
		}
	}

	Ptr<DescriptorExtractor> descriptor = ORB::create();
	//Ptr<DescriptorExtractor> descriptor = BRISK::create();
	Mat descriptors_1, descriptors_2;
	vector<KeyPoint> keypoints1, keypoints2;
	KeyPoint::convert(_pt1, keypoints1);
	KeyPoint::convert(_pt2, keypoints2);
	descriptor->compute(_img1, keypoints1, descriptors_1);
	descriptor->compute(_img2, keypoints2, descriptors_2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<DMatch> match;
	matcher->match(descriptors_1, descriptors_2, match);

	double min_dist = 10000, max_dist = 0;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	//printf("-- Max dist : %f \n", max_dist);
	//printf("-- Min dist : %f \n", min_dist);

	int counter = 0;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist <= 40) {
			counter++;
			pt1_result.push_back(_pt1.at(i));
			pt2_result.push_back(_pt2.at(i));
			ptKF_result.push_back(_ptKF.at(i));

			if (_have3DPts) {
				for (int j = 0; j < size; j++) {
					pt3D_Vec_result[j].push_back(_pt3D_Vec[j].at(i));
				}
			}
		}
	}

	_pt1 = pt1_result;
	_pt2 = pt2_result;
	_ptKF = ptKF_result;
	if (_have3DPts) {
		_pt3D_Vec = pt3D_Vec_result;
	}
}*/

static void
Pts_Re_Check(const Mat &img1, const Mat &img2, vector<Point2f> &pts1, vector<Point2f> &pts2, vector<Point2f> &ptsKF, vector< vector<Point3d> > &pts3D_vec, const bool &have3DPts)
{
	CV_Assert( (pts1.size() == pts2.size()) && (pts1.size() == ptsKF.size()) );

	// --- ORB ---
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	//Ptr<DescriptorExtractor> descriptor = BRISK::create();

	// --- SIFT / SURF ---
	//Ptr<Feature2D> descriptor = xfeatures2d::SIFT::create();
	//Ptr<Feature2D> descriptor = xfeatures2d::SURF::create();
	// -------------------

	Mat descriptors_1, descriptors_2;
	vector<KeyPoint> keypoints1, keypoints2;
	KeyPoint::convert(pts1, keypoints1);
	KeyPoint::convert(pts2, keypoints2);
	descriptor->compute(img1, keypoints1, descriptors_1);
	descriptor->compute(img2, keypoints2, descriptors_2);

	// --- ORB ---
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<DMatch> match;
	matcher->match(descriptors_1, descriptors_2, match);
	// --- ORB---END ---

	// --- SIFT / SURF matcher ---
	//BFMatcher matcher = BFMatcher(NORM_L1);
	//FlannBasedMatcher matcher;
	//vector<DMatch> match;
	//matcher.match(descriptors_1, descriptors_2, match);
	// --- SIFT / SURF matcher---END ---

	double min_dist = 10000, max_dist = 0;
	vector<double> dist_vec;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;

		dist_vec.push_back(dist);
	}
	//printf("-- Max dist : %f \n", max_dist);
	//printf("-- Min dist : %f \n", min_dist);
	//sort(dist_vec.begin(), dist_vec.end());

	sort(dist_vec.begin(), dist_vec.end());
	float median;
	if (dist_vec.size() % 2 == 0) {
		median = (dist_vec[dist_vec.size() / 2 - 1] + dist_vec[dist_vec.size() / 2]) / 2;
	}
	else {
		median = dist_vec[dist_vec.size() / 2];
	}
	float sum = std::accumulate(dist_vec.begin(), dist_vec.end(), 0.0);
	float errors_mean = sum / dist_vec.size();
	float sq_sum = std::inner_product(dist_vec.begin(), dist_vec.end(), dist_vec.begin(), 0.0);
	float errors_stdev = std::sqrt(sq_sum / dist_vec.size() - errors_mean * errors_mean);
	//cout << "Out of "<< dist_vec.size() << "  matched points, those with error < median * 1.5 will survive" << endl;
	//cout << "Average error is " << errors_mean << " +- " << errors_stdev << " (median: " << median << ")"<< endl;

	vector<uchar> status;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < 30) {
		//if (dist < median * 1.5) {
		//if (dist < max(2 * min_dist, 30.0)) {
			status.push_back(1);
		}
		else {
			status.push_back(0);
		}
	}

	// remove bad points.
	remove_bad_points(pts1, status);
	remove_bad_points(pts2, status);
	remove_bad_points(ptsKF, status);
	if (have3DPts)	remove_bad_points(pts3D_vec, status);
}

void 
drawOpticalFlow(const Mat &frame, const vector<Point2f> &pt1, const vector<Point2f> &pt2)
{
	int pt1Size = pt1.size(), pt2Size = pt2.size();
	const int size = min(pt1Size, pt2Size);
	
	for (int i = 0; i < size; i++) {
		line(frame, pt1[i], pt2[i], Scalar(0, 0, 255));
	}
}

bool 
trackKeypoints(Mat &img1, Mat &img2, vector<Point2f> &pt1, vector<Point2f> &pt2, const int &keyFramePtSizeTH)
{
	// For OpticalFlow used
	vector<uchar> status;
	vector<float> err;				
	Size winSize = Size(windowSize, windowSize);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, klt_max_iter, 0.01);

	// optical flow tracking
	calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, err, winSize, maxLevel, termcrit, 0, 0.001);

	// Getting rid of points for which the KLT tracking failed or those who have gone outside the frame
	KLT_Re_Check(pt1, pt2, status);

	if (pt2.size() <= keyFramePtSizeThreshold * keyFramePtSizeTH) {
		return true;
	}
	else {
		return false;
	}
}

bool
trackKeypoints(Mat &img1, Mat &img2, vector<Point2f> &pts1, vector<Point2f> &pts2, vector<uchar> &status, const int &keyFramePtSizeTH)
{
	// For OpticalFlow used
	vector<float> err;
	Size winSize = Size(windowSize, windowSize);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, klt_max_iter, 0.01);

	// optical flow tracking
	calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err, winSize, maxLevel, termcrit, 0, 0.001);

	// Setting as bad correspondences the ones with an extreme outside of the boundary
	vector<float> good_erros;
	for (unsigned int i = 0; i < status.size(); i++) {
		Point2f pt = pts2.at(i);
		if ((pt.x < edgeThreshold) || (pt.y < edgeThreshold) ||
			(pt.x >(img2.cols - edgeThreshold)) || (pt.y >(img2.rows - edgeThreshold))) {

			status.at(i) = 0;
		}
		else {
			good_erros.push_back(err[i]);
		}
	}

	//Setting as bad correspondences the ones with an error bigger than the median*1.5
	sort(good_erros.begin(), good_erros.end());
	float median;
	if (good_erros.size() % 2 == 0) {
		median = (good_erros[good_erros.size() / 2 - 1] + good_erros[good_erros.size() / 2]) / 2;
	}
	else {
		median = good_erros[good_erros.size() / 2];
	}
	float sum = std::accumulate(good_erros.begin(), good_erros.end(), 0.0);
	float errors_mean = sum / good_erros.size();
	float sq_sum = std::inner_product(good_erros.begin(), good_erros.end(), good_erros.begin(), 0.0);
	float errors_stdev = std::sqrt(sq_sum / good_erros.size() - errors_mean * errors_mean);
	//cout << "Out of "<< good_erros.size() << "  matched points, those with error < median * 1.5 will survive" << endl;
	//cout << "Average error is " << errors_mean << " +- " << errors_stdev << " (median: " << median << ")"<< endl;
	for (unsigned int i = 0; i < status.size(); i++) {
		if (status[i] && (err[i] >(median*1.5))) {
			status[i] = 0;
		}
	}

	int ptsSize = countNonZero(status);
	if (ptsSize <= keyFramePtSizeThreshold * keyFramePtSizeTH) {
		return true;
	}
	else {
		return false;
	}
}

bool
trackKeypoints(Mat &img1, Mat &img2, vector<Point2f> &pts1, vector<Point2f> &pts2, vector<Point2f> &ptsKF, const int &keyFramePtSizeTH)
{
	// For OpticalFlow used
	vector<uchar> status;
	vector<float> err;
	Size winSize = Size(windowSize, windowSize);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, klt_max_iter, 0.01);

	// optical flow tracking
	calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err, winSize, maxLevel, termcrit, 0, 0.001);

	// Getting rid of points for which the KLT tracking failed or those who have gone outside the frame
	//KLT_Re_Check(pts1, pts2, ptsKF, status);

	// Setting as bad correspondences the ones with an extreme outside of the boundary
	vector<float> good_erros;
	for (unsigned int i = 0; i < status.size(); i++) {
		Point2f pt = pts2.at(i);
		if ((pt.x < edgeThreshold) || (pt.y < edgeThreshold) ||
			(pt.x > (img2.cols - edgeThreshold)) || (pt.y > (img2.rows - edgeThreshold))) {

			status.at(i) = 0;
		}
		else {
			good_erros.push_back(err[i]);
		}
	}

	//Setting as bad correspondences the ones with an error bigger than the median*1.5
	sort(good_erros.begin(), good_erros.end());
	float median;
	if (good_erros.size() % 2 == 0) {
		median = (good_erros[good_erros.size() / 2 - 1] + good_erros[good_erros.size() / 2]) / 2;
	}
	else {
		median = good_erros[good_erros.size() / 2];
	}
	float sum = std::accumulate(good_erros.begin(), good_erros.end(), 0.0);
	float errors_mean = sum / good_erros.size();
	float sq_sum = std::inner_product(good_erros.begin(), good_erros.end(), good_erros.begin(), 0.0);
	float errors_stdev = std::sqrt(sq_sum / good_erros.size() - errors_mean * errors_mean);
	//cout << "Out of "<< good_erros.size() << "  matched points, those with error < median * 1.5 will survive" << endl;
	//cout << "Average error is " << errors_mean << " +- " << errors_stdev << " (median: " << median << ")"<< endl;
	for (unsigned int i = 0; i < status.size(); i++) {
		if (status[i] && (err[i] >(median*1.5))) {
			status[i] = 0;
		}
	}

	// remove bad points.
	remove_bad_points(pts1, status);
	remove_bad_points(pts2, status);
	remove_bad_points(ptsKF, status);

	//vector< vector<Point3d> > empty;
	//Pts_Re_Check(img1, img2, pts1, pts2, ptsKF, empty, false);

	//if (pts2.size() <= 100) {
	if (pts2.size() <= keyFramePtSizeThreshold * keyFramePtSizeTH) {
		return true;
	}
	else {
		return false;
	}
}

/*bool
trackKeypoints(Mat &img1, Mat &img2, vector<Point2f> &pt1, vector<Point2f> &pt2, vector<Point2f> &ptKF, const int &keyFramePtSizeTH, vector< vector<Point3d> > &points3D_Vec, const bool &have_3DPt)
{
	// For OpticalFlow used
	vector<uchar> status;
	vector<float> err;
	Size winSize = Size(windowSize, windowSize);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, klt_max_iter, 0.01);

	// optical flow tracking
	calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, err, winSize, maxLevel, termcrit, 0, 0.001);

	// Getting rid of points for which the KLT tracking failed or those who have gone outside the frame
	float disMedian;
	if (!have_3DPt) {
		disMedian = KLT_Re_Check(pt1, pt2, ptKF, status, img1.cols, img1.rows);
	}
	else {
		disMedian = KLT_Re_Check(pt1, pt2, ptKF, points3D_Vec, status, img1.cols, img1.rows);
	}

	Pts_Re_Check(img1, img2, pt1, pt2, ptKF, points3D_Vec, have_3DPt, disMedian);

	if (pt2.size() <= keyFramePtSizeThreshold * keyFramePtSizeTH) {
		return true;
	}
	else {
		return false;
	}
}*/

bool
trackKeypoints(Mat &img1, Mat &img2, vector<Point2f> &pts1, vector<Point2f> &pts2, vector<Point2f> &ptsKF, const int &keyFramePtSizeTH, vector< vector<Point3d> > &points3D_Vec, const bool &have_3DPt)
{
	// For OpticalFlow used
	vector<uchar> status;
	vector<float> err;
	Size winSize = Size(windowSize, windowSize);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, klt_max_iter, 0.01);

	// optical flow tracking
	int flags = 0;	// OPTFLOW_LK_GET_MIN_EIGENVALS;
	double minEigThreshold = 1e-3;
	calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err, winSize, maxLevel, termcrit, flags, minEigThreshold);

	// Setting as bad correspondences the ones with an extreme outside of the boundary
	vector<float> good_erros;
	for (unsigned int i = 0; i < status.size(); i++) {
		Point2f pt = pts2.at(i);
		if ((pt.x < edgeThreshold) || (pt.y < edgeThreshold) ||
			(pt.x >(img2.cols - edgeThreshold)) || (pt.y >(img2.rows - edgeThreshold))) {

			status.at(i) = 0; 
		}
		else {
			good_erros.push_back(err[i]);
		}
	}

	//Setting as bad correspondences the ones with an error bigger than the median*1.5
	sort(good_erros.begin(), good_erros.end());
	float median;
	if (good_erros.size() % 2 == 0) {
		median = (good_erros[good_erros.size() / 2 - 1] + good_erros[good_erros.size() / 2]) / 2;
	}
	else {
		median = good_erros[good_erros.size() / 2];
	}
	float sum = std::accumulate(good_erros.begin(), good_erros.end(), 0.0);
	float errors_mean = sum / good_erros.size();
	float sq_sum = std::inner_product(good_erros.begin(), good_erros.end(), good_erros.begin(), 0.0);
	float errors_stdev = std::sqrt(sq_sum / good_erros.size() - errors_mean * errors_mean);
	//cout << "Out of "<< good_erros.size() << "  matched points, those with error < median * 1.5 will survive" << endl;
	//cout << "Average error is " << errors_mean << " +- " << errors_stdev << " (median: " << median << ")"<< endl;
	for (unsigned int i = 0; i < status.size(); i++) {
		if (status[i] && (err[i] >(median*1.5))) {
			status[i] = 0;
		}
	}

	// remove bad points.
	remove_bad_points(pts1, status);
	remove_bad_points(pts2, status);
	remove_bad_points(ptsKF, status);
	if (have_3DPt)	remove_bad_points(points3D_Vec, status);

	//Pts_Re_Check(img1, img2, pts1, pts2, ptsKF, points3D_Vec, have_3DPt);

	//if (pts2.size() <= 100) {
	if (pts2.size() <= keyFramePtSizeThreshold * keyFramePtSizeTH) {
		return true;
	}
	else {
		return false;
	}
}