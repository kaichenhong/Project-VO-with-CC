#include "triangulate.h"

static Point2f 
pixel2cam(const float &x, const float &y, const Mat& K)
{
    return Point2f
    (
        ( x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}

void 
normalizePoints(const Mat &_points4D, vector<Point3d> &_points3D)
{
	for (int i = 0; i < _points4D.cols; i++) {
		Mat x = _points4D.col(i);
		x /= x.at<float>(3, 0);
		Point3d p(
			x.at<float>(0, 0),
			x.at<float>(1, 0),
			x.at<float>(2, 0)
		);
		_points3D.push_back(p);
	}
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

void 
showTriangulation(
	const vector<Point2f> &keypoint_1,
	const vector<Point2f> &keypoint_2,
	const Mat &R, const Mat &t, 
	const vector<Point3d> &points)
{
	//-- 验证三角化点与特征点的重投影关系
	//Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	//Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);

	for (int i = 0; i < keypoint_1.size(); i++)
	{
		Point2d pt1_cam = pixel2cam(keypoint_1[i].x, keypoint_1[i].y, K);
		Point2d pt1_cam_3d(
			points[i].x / points[i].z,
			points[i].y / points[i].z
		);

		cout << "point in the first camera frame: " << pt1_cam << endl;
		cout << "point projected from 3D " << pt1_cam_3d << ", d=" << points[i].z << endl;

		// 第二个图
		Point2f pt2_cam = pixel2cam(keypoint_2[i].x, keypoint_2[i].y, K);
		Mat pt2_trans = R*(Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
		pt2_trans /= pt2_trans.at<double>(2, 0);
		cout << "point in the second camera frame: " << pt2_cam << endl;
		cout << "point reprojected from second frame: " << pt2_trans.t() << endl;
		cout << endl;
	}
}

/*void 
triangulation ( 
    const vector<Point2f> &keypoint_1,
    const vector<Point2f> &keypoint_2,
    const Mat &R, const Mat &t, 
    vector<Point3d> &points)
{
    Mat T1 = (Mat_<float>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );
    
    //Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1 );
    
    vector<Point2f> pts_1, pts_2;
    for (int i = 0; i < keypoint_1.size(); i++) {
        // 将像素坐标转换至相机坐标
        pts_1.push_back ( pixel2cam( keypoint_1[i].x, keypoint_1[i].y, K) );
        pts_2.push_back ( pixel2cam( keypoint_1[i].x, keypoint_1[i].y, K) );
    }
    
    Mat pts_4d;
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
    
    // 转换成非齐次坐标
    for (int i = 0; i < pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
        points.push_back( p );
    }
}*/

void 
triangulate_Points(const Mat &P1, const Mat &P2, const vector<Point2f> &pts1, const vector<Point2f> &pts2, vector<uchar> &status, vector<Point3d> &pts3D)
{
	CV_Assert(pts1.size() == pts2.size());
	//--- copy pts and remove bad points
	vector<Point2f> pts1_temp = pts1, pts2_temp = pts2;
	remove_bad_points(pts1_temp, status);
	remove_bad_points(pts2_temp, status);
	
	//--- triangulate
	Mat points4D;
	triangulatePoints(P1, P2, pts1_temp, pts2_temp, points4D);

	//--- normalize 4D points to 3D points and save to pt3DVec
	vector<Point3d> points3D;
	normalizePoints(points4D, points3D);

	//--- Chierality check
	vector<uchar> status_temp;
	for (unsigned int i = 0; i < points3D.size(); i++)
		status_temp.push_back((points3D[i].z > 0) ? 1 : 0);

	cout << "triangulation likes " << countNonZero(Mat(status_temp))
		 << " out of " << pts2_temp.size()
		 << " (" << (float)(countNonZero(Mat(status_temp))) / (float)(pts2_temp.size()) * 100 << "%)" << endl << endl;

	//--- re-build status
	status = rebuildStatus(status, status_temp);
	CV_Assert(pts1.size() == status.size());

	//--- re-build pts3D
	const int N = status.size();
	int k = 0;
	for (int i = 0; i < N; i++) {
		if (status[i] == 1) {
			pts3D.push_back(points3D[k]);
			k++;
		}
		else {
			pts3D.push_back( Point3d(0, 0, 0) );
		}
	}
	CV_Assert(pts1.size() == pts3D.size());
}