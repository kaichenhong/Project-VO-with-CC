#define _CRT_SECURE_NO_WARNINGS
#include "frame.h"

struct Frame *
getFrame(int frameNum)
{
    struct Frame *frame = (struct Frame *)calloc(1, sizeof(struct Frame));

    frame->id = frameNum;

    char fileName[100];
    sprintf(fileName, fileLocation, frameNum);
    frame->img = imread(fileName);

    cvtColor(frame->img, frame->grayImg, CV_BGR2GRAY);

    return frame;
}

bool 
getFrame(struct Frame &frame, int frameNum) 
{
	frame.id = frameNum;

	char fileName[100];
	sprintf(fileName, fileLocation, frameNum);
	frame.img = imread(fileName);
	if (frame.img.empty()) {
		return false;
	}

	if (frame.img.type() != CV_8UC1) {
		cvtColor(frame.img, frame.grayImg, CV_BGR2GRAY);
	}
	else {
		frame.grayImg = frame.img;
	}

	return true;
}