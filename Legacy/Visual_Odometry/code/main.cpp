#include <iostream>
#include "debug.h"
#include "odometry.h"
#include "devkit\cpp\evaluate_odometry.h"

//#include <direct.h>
//#define GetCurrentDir _getcwd

using namespace std;
using namespace cv;

int main()
{
	//char cCurrentPath[FILENAME_MAX];
	//GetCurrentDir(cCurrentPath, sizeof(cCurrentPath));
	//bool success = eval();
	//showHammingDistance();
	
	//debug();
	//odometryRun();
	
	//odometryRunLBP();
	//odometryRunLBP_Farneback();
	
	odometryRun_PNP();
	//odometryRun_ALL_PNP();
	//odometryRun_ALL_PNP2();
	//odometryRun_KF();		// have error
	//odometryRun_PNP_Test();
	
	//odometryRun_PNP2();

	//testingAndCheck();

	/*---- run evaluate odometry ----*/
	system("pause");
	bool success = eval();


	//waitKey(0);
	system("pause");
    return 0;
}