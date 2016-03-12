/**
 *
 * Copyright 2015 by Abhinav Jain <abhinavjain241@gmail.com>
 *
 * Code for extracting 'red' coloured plots in the graph.
 * Creates a file called binarized_red_extracted.png which is further processed by another code which returns coordinates.
 * TODO: Extend to other colours, and fuse both the functionalities.
 */


#include "opencv2/opencv.hpp"

int main( int argc, char** argv ) {
	// Read image bgr_image - The original BGR image
	cv::Mat bgr_image = cv::imread( "../../Images/p002.png"); // RGB right now
	cv::Mat orig_image = bgr_image.clone(); // Clone the original image
	cv::medianBlur(bgr_image, bgr_image, 3); // Apply median blur
	cv::Mat hsv_image; // Create a HSV image template
	cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV); // Convert RGB --> HSV
	// Threshold the HSV image, keep only the red pixels
	cv::Mat lower_red_hue_range;
	cv::Mat upper_red_hue_range;
	cv::inRange(hsv_image, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
	cv::inRange(hsv_image, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);
	// Combine the above two images
	cv::Mat red_hue_image;
	cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	cv::GaussianBlur(red_hue_image, red_hue_image, cv::Size(9, 9), 2, 2);
	// Now red_hue_image is the color-extracted image
	cv::Mat binary;
	cv::threshold(red_hue_image, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imwrite("red_extracted.png", red_hue_image);	
	imwrite("binarized_red_extracted.png", binary);
	cv::namedWindow("Threshold lower image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Threshold lower image", lower_red_hue_range);
	cv::namedWindow("Threshold upper image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Threshold upper image", upper_red_hue_range);
	cv::namedWindow("Combined threshold images", cv::WINDOW_AUTOSIZE);
	cv::imshow("Combined threshold images", red_hue_image);
	cv::namedWindow("Detected red circles on the input image", cv::WINDOW_AUTOSIZE);	
	cv::imshow("Detected red circles on the input image", orig_image);
	cv::waitKey(0);
	// Setup SimpleBlobDetector parameters.
	cv::SimpleBlobDetector::Params params;
	// Change thresholds
	params.minThreshold = 5;
	params.maxThreshold = 200;
	// Filter by Area.
	params.filterByArea = false;
	params.minArea = 1500;
	// Filter by Circularity
	params.filterByCircularity = false;
	params.minCircularity = 0.1;
	// Filter by Convexity
	params.filterByConvexity = false;
	params.minConvexity = 0.87;
	// Filter by Inertia
	params.filterByInertia = false;
	params.minInertiaRatio = 0.01;
	// Storage for blobs
	std::vector<cv::KeyPoint> keypoints;

#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2
	// Set up detector with params
	cv::SimpleBlobDetector detector(params);
	// Detect blobs
	detector.detect( red_hue_image, keypoints);
#else 
	// Set up detector with params
	Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);   
	// Detect blobs
	detector->detect( red_hue_image, keypoints);
#endif 
	cv::Mat im_with_keypoints;
	cv::drawKeypoints( upper_red_hue_range, keypoints, im_with_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	// Show blobs
	cv::imshow("keypoints", im_with_keypoints );
	cv::waitKey(0);
}

