#ifndef CROSSINGNUMBER_H
#define CROSSINGNUMBER_H

#include "extraction_config.h"

class CrossingNumber : public QObject
{
    Q_OBJECT

public:
    CrossingNumber();

    void setParams(const cv::Mat &imgSkeleton, const PREPROCESSING_RESULTS &input);
    void findMinutiae();
    void clean();

    void findMinutiaeInBatch(QVector<cv::Mat> skeletons,QVector<cv::Mat> oMap);
    double getBatchTime();

    QVector<MINUTIA> getMinutiae() const;
    QVector<QVector<MINUTIA>> getMinutiaeMap();
    QVector<MINUTIA> getMinutiaeFromMap(int index);
private:

    cv::Mat imgSkeleton;
    PREPROCESSING_RESULTS input;

    QVector<QVector<MINUTIA>> minutiaeMap;
    QVector<MINUTIA> minutiae;
    double batchCNTime;
    af::array findInSingleSkeleton(af::array skeleton);
    QVector<MINUTIA> matrixToVector(af::array CN,cv::Mat oMap);
};

#endif // CROSSINGNUMBER_H
