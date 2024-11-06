

#include<iostream>
#include<opencv2/opencv.hpp>
#include<chrono>
#include<cstring> //for size calculation

void preprocess_gray_2(const std::string &image_path,const cv::Size &target_size,std::vector<float> &image_final, float &image_size_kb)
{
	//load the image in color in 3 channels
	auto start=std::chrono::high_resolution_clock::now();
	cv::Mat image=cv::imread(image_path,cv::IMREAD_COLOR);

	//check if the image is loaded successfully
	if(image.empty()){
		std::cerr<<"Error loading the image"<<std::endl;
		return ;
	}
	auto end=std::chrono::high_resolution_clock::now();

	std::chrono::duration<double>duration=end-start;
        std::cout<<"Image loaded from disc in:"<<duration.count()<<"s"<<std::endl;

	start=std::chrono::high_resolution_clock::now();
	//Resize the image to target dimension
	cv::Mat resized_image;
	cv::resize(image,resized_image,target_size);

	//convert image to grayscale
	cv::Mat gray_image;
	cv::cvtColor(resized_image,gray_image,cv::COLOR_BGR2GRAY);

	//Normalize the iamge to (0,1)
	gray_image.convertTo(gray_image,CV_32F,1.0/255.0);

	//get the image size in KB
	image_size_kb=image.total()*image.elemSize()/1024.0;

	//stack the grayscale image into the first channel of a blank 3D array (1,3,height,width)
	image_final.clear(); //clear any previous data
	image_final.resize(1*3*target_size.height*target_size.width,0.0f);

	//populate the first channel with the grayscale data
	int idx=0;
	for (int i=0;i<gray_image.rows;++i){
		for (int j=0;j<gray_image.cols;++j){
			image_final[idx++]=gray_image.at<float>(i,j);
		}
	}

	end=std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double>duration=end-start;
	duration=end-start;
        std::cout<<"Image preprocessed in:"<<duration.count()<<"s"<<std::endl;

}




std::vector<uchar> load_image_to_buffer(const std::string& image_path) {
	//read the image from the disc
	cv::Mat image =cv::imread(image_path,cv::IMREAD_COLOR);

	if (image.empty()){
		std::cerr<<"Error loading the image"<<std::endl;
		return std::vector<uchar>(); //return the empty buffer if failed to load the image
	}

	//vector to hodl the image buffer (using hte PNG compression)
	std::vector<uchar> buffer;

	//compress the image into buffer using the PNG
        //std::vector<int> compression_params={cv::IMWRITE_PNG_QUALITY,90};
	bool success =cv::imencode(".png",image,buffer);

	if (!success){
		std::cerr<<"Error encoding the image to the buffer"<<std::endl;
		return std::vector<uchar>();
	}

	return buffer; //returns the image encoded as the PNG buffer 
}







void preprocess_gray_3(const std::vector<uchar> &png_buffer, const cv::Size &target_size, std::vector<float> &image_final, float &image_size_kb)
{
    // Start timing the image loading and decoding process
    auto start = std::chrono::high_resolution_clock::now();

    // Decode the image from the PNG buffer
    cv::Mat image = cv::imdecode(png_buffer, cv::IMREAD_COLOR);  // Decode PNG to color image (3 channels)

    // Check if the image is loaded successfully
    if (image.empty()) {
        std::cerr << "Error decoding the PNG image!" << std::endl;
        return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Image decoded from PNG buffer in: " << duration.count() << "s" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    // Resize the image to target dimensions
    cv::Mat resized_image;
    cv::resize(image, resized_image, target_size);

    // Convert image to grayscale
    cv::Mat gray_image;
    cv::cvtColor(resized_image, gray_image, cv::COLOR_BGR2GRAY);

    // Normalize the image to (0, 1)
    gray_image.convertTo(gray_image, CV_32F, 1.0 / 255.0);

    // Get the image size in KB
    image_size_kb = image.total() * image.elemSize() / 1024.0;

    // Stack the grayscale image into the first channel of a blank 3D array (1, 3, height, width)
    image_final.clear(); // Clear any previous data
    image_final.resize(1 * 3 * target_size.height * target_size.width, 0.0f);

    // Populate the first channel with the grayscale data
    int idx = 0;
    for (int i = 0; i < gray_image.rows; ++i) {
        for (int j = 0; j < gray_image.cols; ++j) {
            image_final[idx++] = gray_image.at<float>(i, j);
        }
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Image preprocessed in: " << duration.count() << "s" << std::endl;
}






























		

int main()
{
	//load the image from the file path
	std::string image_path="image.png";
	

	//image loaded into the image buffer 
	std::vector<uchar> image_buffer=load_image_to_buffer(image_path);
	if (!image_buffer.empty()){
		std::cout<<"image loaded successgully"<<image_buffer.size()/1024.0<<"KB"<<std::endl;
	}
	else{
		std::cout<<"failed to load the image"<<std::endl;
	}
	


	cv::Size target_size(2464,3280); //set the target size

	std::vector<float> img_final; //declare the variable of the final image
	float image_size_kb; //declare the compressed image size


	auto start=std::chrono::high_resolution_clock::now();

	//this preprocessing is done on the image in the buffer format
	preprocess_gray_3(image_buffer,target_size,img_final,image_size_kb);

	auto end=std::chrono::high_resolution_clock::now();

	std::chrono::duration<double>duration=end-start;
	std::cout<<"image preprocess runtime:"<<duration.count()<<"s"<<std::endl;

	//output the size of the image in KB
	std::cout<<"Image size:"<<image_size_kb<<"KB"<<std::endl;

	//optionally print the first few lines for debugging
	//std::cout<<"First few elements of img_final"<<std::endl;

	//for(int i=0;i<10;++i){
	//	std::cout<<img_final[i]<<" ";
	//}
	//std::cout<<std::endl;

	return 0;

}
