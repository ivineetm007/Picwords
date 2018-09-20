
% Mean shift Segmentation code is taken from 
% Alireza Asvadi
% Department of ECE, SPR Lab
% Babol (Noshirvani) University of Technology
% http://www.a-asvadi.ir
% 2013
RESTRICTION
Image should have plain background.
------------------------------------------------------------------------------------------
TEXT GENERATION:
HOW TO RUN
Step1: Copy and paste the description paragraph of the person in 'textfile.txt'
Step2: Run keywords.py
	In windows ,
		->just double click 'keywords.py'  If python is installed and added in environment variable.
		or
		->open cmd in the same directory and run as
		python keywords.py
PATCH GENERATION AND TEXT WARPING IN MATLAB
Step1:	Open Demo.m in matlab and run the code. //Matlab version should be 2016 or later. Some inbuilt functions are not in older versions. 
Step2:	In Demo.m ,  . Enter the name of image on which you want to test .  //copy the image in the same folder.
Step3:	The program will take user input . Press 1 If given image is of full body of a person and Press 0 If given image is of face of a person.
Step4: A Segmented image will be displayed . Click on image to proceed further. 
-----------------------------------------------------------------------------------------
REFERENCES
1.PicWords: Render a Picture by Packing Keywords
Zhenzhen Hu, Si Liu, Member, IEEE, Jianguo Jiang, Richang Hong, Member, IEEE, MengWang, Member, IEEE,
and Shuicheng Yan, Senior Member, IEEE
2.PicWords: Creating Pictures by its keywords
Ankur Singh1, Shruti Garg2
	1 M.Tech, Department of Computer Science and Engineering, Birla Institute of Technology, Mesra, Ranchi, India.
	2 Assistant Professor, Department of Computer Science and Engineering, Birla Institute of Technology, Mesra,
	Ranchi, India.
