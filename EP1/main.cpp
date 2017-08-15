#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "iostream"

using namespace cv;
using namespace std;

int main(int, char**){
  VideoCapture cap("vid4.avi");
  //cap.open("vid4.avi");
  bool bSuccess = cap.read(NextFrameBGR);

    



  return 0;
}