#include "crossingnumber.h"

CrossingNumber::CrossingNumber()
{

}

void CrossingNumber::setParams(const cv::Mat &imgSkeleton, const PREPROCESSING_RESULTS &input)
{
    this->imgSkeleton = imgSkeleton;
    this->input = input;
}

void CrossingNumber::findMinutiae()
{
    int cn;
    int quality;

    for (int y = 0; y < this->imgSkeleton.rows; y++) {
        for (int x = 0; x < this->imgSkeleton.cols; x++) {
            if (this->imgSkeleton.at<uchar>(y, x) == 0) {
                cn = abs((this->imgSkeleton.at<uchar>(y - 1, x - 1)) - (this->imgSkeleton.at<uchar>(y, x - 1))) +
                        abs((this->imgSkeleton.at<uchar>(y, x - 1)) - (this->imgSkeleton.at<uchar>(y + 1, x - 1))) +
                        abs((this->imgSkeleton.at<uchar>(y + 1, x - 1)) - (this->imgSkeleton.at<uchar>(y + 1, x))) +
                        abs((this->imgSkeleton.at<uchar>(y + 1, x)) - (this->imgSkeleton.at<uchar>(y + 1, x + 1))) +
                        abs((this->imgSkeleton.at<uchar>(y + 1, x + 1)) - (this->imgSkeleton.at<uchar>(y, x + 1))) +
                        abs((this->imgSkeleton.at<uchar>(y, x + 1)) - (this->imgSkeleton.at<uchar>(y - 1, x + 1))) +
                        abs((this->imgSkeleton.at<uchar>(y - 1, x + 1)) - (this->imgSkeleton.at<uchar>(y - 1, x))) +
                        abs((this->imgSkeleton.at<uchar>(y - 1, x)) - (this->imgSkeleton.at<uchar>(y - 1, x - 1)));

                if (this->input.qualityMap.cols > x && this->input.qualityMap.rows > y) quality = this->input.qualityMap.at<uchar>(y,x);
                else quality = 254;

                if (cn / 255 / 2 == 1) {
                    this->minutiae.push_back(MINUTIA{QPoint{x,y}, 0, this->input.orientationMap.at<float>(y,x) + M_PI_2, quality, QPoint{this->input.imgOriginal.cols, this->input.imgOriginal.rows}});
                }
                else if (cn / 255 / 2 == 3) {
                    this->minutiae.push_back(MINUTIA{QPoint{x,y}, 1, this->input.orientationMap.at<float>(y,x) + M_PI_2, quality, QPoint{this->input.imgOriginal.cols, this->input.imgOriginal.rows}});
                }
            }
        }
    }
}

QVector<MINUTIA> CrossingNumber::getMinutiae() const
{
    return minutiae;
}

void CrossingNumber::clean()
{
    this->minutiae.clear();
    this->minutiaeMap.clear();
    this->batchCNTime=0;
}

af::array CrossingNumber::findInSingleSkeleton(af::array skeleton){
    skeleton=skeleton.as(u8)/255;
    af::array MinutiaMatrix=af::constant(0,skeleton.dims(0),skeleton.dims(2));

    for (int x=1;x<skeleton.dims(1)-1;x++) {
        for (int y=1;y<skeleton.dims(0)-1;y++) {
            MinutiaMatrix(y,x)=((
                                       af::abs(skeleton(y-1,x-1)-skeleton(y,x-1))+
                                       af::abs(skeleton(y,x-1)-skeleton(y+1,x-1))+
                                       af::abs(skeleton(y+1,x-1)-skeleton(y+1,x))+
                                       af::abs(skeleton(y+1,x)-skeleton(y+1,x+1))+
                                       af::abs(skeleton(y+1,x+1)-skeleton(y,x+1))+
                                       af::abs(skeleton(y,x+1)-skeleton(y-1,x+1))+
                                       af::abs(skeleton(y-1,x+1)-skeleton(y-1,x))+
                                       af::abs(skeleton(y-1,x)-skeleton(y-1,x-1))
                                        )/2)*skeleton(y,x);
        }
    }
   return MinutiaMatrix.as(u8);
}

void CrossingNumber::findMinutiaeInBatch(QVector<cv::Mat> skeletons,QVector<cv::Mat> oMap){
    //prevod na array
    af::array matrix(skeletons[0].rows,skeletons[0].cols,skeletons.size()); // array of skeletons
    for (int i=0;i<skeletons.size();i++) {
        cv::Mat helperMat;
        cv::transpose(skeletons[i],helperMat);
        matrix(af::span,af::span,i)=af::array(skeletons[i].rows,skeletons[i].cols,helperMat.data).as(u8);
    }//created array from QVector

    //gfor + meranie casu
    af::timer::start();
    gfor(af::seq k,matrix.dims(2)){
        matrix(af::span,af::span,k)=this->findInSingleSkeleton(matrix(af::span,af::span,k));
    }//vyhladanie markantov v obraze
    this->batchCNTime=af::timer::stop();

    for (int i=0;i<matrix.dims(2);i++) {
        this->minutiaeMap.push_back(this->matrixToVector(matrix(af::span,af::span,i),oMap[i]));
    }//najdene markanty do QVector -> minutiaeMap
}

double CrossingNumber::getBatchTime(){
    return this->batchCNTime;
}

QVector<MINUTIA> CrossingNumber::getMinutiaeFromMap(int index){
    QVector<MINUTIA> minutiae;
    if(index<minutiaeMap.size() && index>0)
        return this->minutiaeMap[index];
    return minutiae;
}

QVector<MINUTIA> CrossingNumber::matrixToVector(af::array CN, cv::Mat oMap){
    //array to cv::Mat
    uchar* data = CN.as(u8).T().host<uchar>();
    cv::Mat Map = cv::Mat((int)CN.dims(0), (int)CN.dims(1),CV_8UC1, data).clone();
    af::freeHost(data);


    QVector<MINUTIA> minutiae;
    //search mat for minutiae
    for (int y=0;y<Map.rows;y++) {
        for (int x=0;x<Map.cols;x++) {
            if(Map.at<uchar>(y,x) == 1)
                minutiae.push_back(MINUTIA{QPoint{x,y},0,oMap.at<float>(y,x)+M_PI_2,254,QPoint{Map.cols,Map.rows}});
            else if(Map.at<uchar>(y,x) == 3)
                minutiae.push_back(MINUTIA{QPoint{x,y},1,oMap.at<float>(y,x)+M_PI_2,254,QPoint{Map.cols,Map.rows}});
        }
    }
    return minutiae;
}
