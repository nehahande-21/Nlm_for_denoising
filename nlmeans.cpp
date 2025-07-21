#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

Mat enhanceUnderwaterImage(const Mat& img) {
    Mat lab;
    cvtColor(img, lab, COLOR_BGR2Lab);

    vector<Mat> lab_planes(3);
    split(lab, lab_planes);

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);

    Mat enhanced;
    merge(lab_planes, lab);
    cvtColor(lab, enhanced, COLOR_Lab2BGR);

    return enhanced;
}

void denoiseImage(const string& inputPath, const string& outputPath) {
    cout << "📂 Loading image: " << inputPath << endl;
    Mat img = imread(inputPath);
    if (img.empty()) {
        cerr << "❌ Error: Unable to load image." << endl;
        return;
    }

    Mat enhanced = enhanceUnderwaterImage(img);
    cout << "📈 Starting denoising..." << endl;

    Mat denoised;
    auto start = high_resolution_clock::now();
    fastNlMeansDenoisingColored(enhanced, denoised, 10, 10, 7, 21);
    auto end = high_resolution_clock::now();

    double duration = duration_cast<milliseconds>(end - start).count();
    cout << "✅ Denoising completed in " << duration / 1000.0 << " seconds." << endl;

    imwrite(outputPath, denoised);
    cout << "🖼️ Denoised image saved to: " << outputPath << endl;

    imshow("Denoised Output", denoised);
    waitKey(0);
    destroyAllWindows();
}

int main() {
    string inputImage = "noisy_input.jpeg";
    string outputImage = "denoised_output.jpeg";

    denoiseImage(inputImage, outputImage);
    return 0;
}
