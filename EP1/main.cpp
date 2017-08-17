#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "iostream"

using namespace cv;
using namespace std;

int main(int, char**){
  VideoCapture cap("vid4.avi");
  if(!cap.isOpened()){
    cout << "Cannot open the video file" << endl;
    return -1;
  }

  double count = cap.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
  cap.set(CV_CAP_PROP_POS_FRAMES, count+3); //Set index to last frame
  namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);

  Mat frame;
  cap.read(frame);

  while(1){
    bool success = cap.read(frame);
    if (!success){
      cout << "Cannot read frame " << endl;
      break;
    }
    imshow("MyVideo", frame);
    if(waitKey(0) == 27) break;
  }
  return 0;
}