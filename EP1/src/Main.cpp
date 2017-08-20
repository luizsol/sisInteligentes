#include <opencv2/opencv.hpp>
#include "ImgProcessor.h"
#include "iostream"

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  ImgProcessor processor("img/img0105.bmp");
  processor.showImage("A imagem");

  return 0;
}