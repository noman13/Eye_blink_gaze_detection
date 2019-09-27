% Tracking by recognizing faces (using Kanade-Lucas-Tomasi (KLT))
% Detect closed eyes from position of face
% We use mechanics SVM (Support Vector Machine) for inspection
% Statistics Toolbox is required

%% initialize
clear all; close all; clc; imaqreset;

%% Image preservation folder
imdir = 'dbim';
[~,~,~] = rmdir(imdir,'s');
imdirClosed = [imdir filesep 'closed'];
imdirOpened = [imdir filesep 'opened'];
[~,~,~] = mkdir(imdir);
[~,~,~] = mkdir(imdirClosed);
[~,~,~] = mkdir(imdirOpened);

%% Video aquisition (Object Definition)
cam = webcam();


right=imread('RIGHT.jpg');
left=imread('LEFT.jpg');
straight=imread('STRAIGHT.jpg');

% Display object
videoPlayer = vision.DeployableVideoPlayer();
videoPlayerZoom = vision.DeployableVideoPlayer();

%face and eye detector
faceDetector = vision.CascadeObjectDetector(); 
eyesDetector = vision.CascadeObjectDetector('EyePairBig'); 

% Point Tracker Object (Tracking Face)
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);


runLoop = true;
numPts = 0;
frameCount = 0;
eyeFlag = false;
eyeCount =  int32(0);
time = 0;
numClosed = 0;
numOpened = 0;
isCaptureClosed=false;
isCaptureOpened=false;
isTrain=false;

%% Capture/Train
sz = get(0,'ScreenSize');
figure('MenuBar','none','Toolbar','none','Position',[1100 sz(4)-500 150 260])
uicontrol('Style', 'pushbutton', 'String', 'CAPTURE OPEN',...
    'Position', [20 160 120 80],...
    'Callback', 'isCaptureOpened=true;');
uicontrol('Style', 'pushbutton', 'String', 'CAPTURE CLOSE',...
	'Position', [20 70 120 80],...
    'Callback', 'isCaptureClosed=true;');
uicontrol('Style', 'pushbutton', 'String', 'TRAIN',...
    'Position', [20 20 120 40],...
    'Callback', 'isTrain=true;');
%k = figure('Position', [1000 30 300 500]);
%training image display
h = figure('Position', [20 30 500 500]);
subplot(2,2,1);
set(gca,'xtick',[],'ytick',[],'Xcolor','w','Ycolor','w')
title('Positive Class(Eyes Closed)','FontSize',14); 
subplot(2,2,3);
set(gca,'xtick',[],'ytick',[],'Xcolor','w','Ycolor','w')
title('Negative Class(Eyes Opened)','FontSize',14);


while (runLoop)
    %capture image data
    videoFrameOrg = snapshot(cam);
    
    videoFrame = videoFrameOrg;
    videoFrameGray = rgb2gray(videoFrameOrg);
    
    % Tracking points 
    if numPts < 10
        % Face Detection
        bbox = faceDetector.step(videoFrameGray);
        
        if ~isempty(bbox)
            % Inspect the tracker point from the detected area
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
            
            % Point Tracker
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);
            
            % save points
            oldPoints = xyPoints;
            
            % Convert to co-ordinator
            bboxPoints = bbox2points(bbox(1, :));
            
            % Convert to a vector of the form shown on the right [x1 y1 x2 y2 x3 y3 x4 y4]
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            % Display boundary area
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            % Display detected corners
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end
        
    else
        %Tracking Mode
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
        
        numPts = size(visiblePoints, 1);
        
        if numPts >= 10
            % Geometric transformation of old points
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            
            % Convert to a ticket using a row
            bboxPoints = transformPointsForward(xform, bboxPoints);
            
            % Convert to a vector of the form shown on the right [x1 y1 x2 y2 x3 y3 x4 y4]
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            %  Display boundary area
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            % Display detected corners
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
            
            % reset points
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
            
        end
        
        % Fitting the face part into a square (for both eye examination)
        zoomPoints = [1 1; 200 1; 200 200; 1 200];    
        T = fitgeotrans(bboxPoints, zoomPoints,'projective');
        rectifiedFace = imwarp(videoFrameOrg, T, 'OutputView', imref2d([200 200]));
        
        % Detect both eyes from the image of the face
        bboxEyes = step(eyesDetector, rectifiedFace);
        
        if ~isempty(bboxEyes) % Was I able to find both eyes?
            % Fit the region of both eyes detected into a rectangle
            bboxEyesPoints = bbox2points(bboxEyes(1,:));
            zoomPoints = [1 1; 440 1; 440 100; 1 100];  
            T = fitgeotrans(bboxEyesPoints, zoomPoints,'projective');
            videoFrameEyes = imwarp(rectifiedFace, T, 'OutputView', imref2d([100 440]));
            
            if (isCaptureClosed || isCaptureOpened)
                % Save closed eyes to file
                if (isCaptureClosed)
                    numClosed = numClosed+1;
                    imwrite(videoFrameEyes, [imdirClosed filesep 'frame' num2str(numClosed) '.png']);
                end
                
                % Save open eyes in a file
                if (isCaptureOpened)
                    numOpened = numOpened+1;
                    imwrite(videoFrameEyes, [imdirOpened filesep 'frame' num2str(numOpened) '.png']);
                end
                
                % Specify a set of training images
                posSets = imageSet(fullfile(imdir,'closed'));
                negSets = imageSet(fullfile(imdir,'opened'));
                figure(h);
                subplot(2,2,1);
                if ~isempty(posSets.ImageLocation)
                    montage(posSets.ImageLocation);% pos image
                end
                title('Positive Class(Eyes Closed)','FontSize',14); 
                subplot(2,2,3);
                if ~isempty(negSets.ImageLocation)
                    montage(negSets.ImageLocation);% neg image
                end
                title('Negative Class(Eyes Opened)','FontSize',14); 
                isCaptureClosed = false;
                isCaptureOpened = false;
            end
            
            
            if exist('svmModel','var')
                img = imresize(videoFrameEyes, [20 88]);       % Changing image size
                % Extract HOG features
                testFeatures = extractHOGFeatures(img,'CellSize',cellSize);
                % Predictions by passing features to classifiers
                preEyeFlag = eyeFlag;
                eyeFlag = predict(svmModel, testFeatures);
                if ~preEyeFlag && eyeFlag
                    eyeCount = eyeCount + 1;
                end
            else
                preEyeFlag = false;
            end
            
            % Display boundary showing detected eye pupil 
            rectifiedFace = insertShape(rectifiedFace, 'Rectangle', bboxEyes, 'LineWidth', 3, 'Color', 255*[eyeFlag ~eyeFlag 0]);
        end
        
        
        if isTrain
            isTrain = false;
            
            % Specify a set of training images
            trainingSets = imageSet(imdir, 'recursive');
            
            % Show training image set
            figure(h);
            subplot(2,2,1);montage(trainingSets(1).ImageLocation);title('positive','FontSize',14);   % pos image 
            subplot(2,2,3);montage([trainingSets(2:end).ImageLocation]);title('negative','FontSize',14); % neg image
            
            % labeling pos image as true, neg image as false
            trainingLabels    = false(sum([trainingSets.Count]),1);
            trainingLabels(1:trainingSets(1).Count) = true; %Specify (closed folder) as pos
            trainingLabels(trainingSets(1).Count+1:sum([trainingSets(2:end).Count])) = false; %Specify (open folder) as neg
            
            % Use cell size of 4x4
            cellSize = [4 4];
            
            % In order to calculate hogFeatureSize in advance, only one HOG extraction
            img = read(trainingSets(1), 1);
            img = imresize(img, [20 88]);
            [hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',cellSize);
            hogFeatureSize = length(hog_4x4);
            
            % trainingFeatures Prearrange the array to store
            trainingFeatures  = zeros(sum([trainingSets.Count]),hogFeatureSize,'single');
            
            % Extract HOG feature value from all tracing image
            k = 1;
            for index = 1:numel(trainingSets)   % 1=>pos, 2=>neg
                for i = 1:trainingSets(index).Count
                    img = read(trainingSets(index), i);  % Reading of training images
                    img = imresize(img, [20 88]);
                    trainingFeatures(k,:) = extractHOGFeatures(img,'CellSize',cellSize); % Feature amount extraction
                    k = k+1;
                end
            end
            
            % Support for classifier of support vector machine (SVM) (for fitcsvm () function)
            svmModel = fitcsvm(trainingFeatures, trainingLabels);
        end
        
        % Display the screen
        step(videoPlayerZoom, rectifiedFace);
    end
    
    % Model display
    if exist('svmModel', 'var')
        % Displaying blink detection mode
        videoFrame = insertText(videoFrame, [0 0], 'Running eyeblink detection', 'FontSize', 40, 'BoxColor', [30 180 160]);
        % Eye close count
        videoFrame = insertText(videoFrame, [0 50],  ['Eye Closed ' num2str(eyeCount,'%d') ' times'], 'FontSize', 20, 'BoxColor', [140 30 180]);
    else
        % Displaying not trained
        videoFrame = insertText(videoFrame, [0 0], 'Not Trained', 'FontSize', 40, 'BoxColor', [180 150 30]);
    end
    
    
    step(videoPlayer, videoFrame);
    
    % Set runLoop to false when the video player is closed
    runLoop = isOpen(videoPlayer);
    
    drawnow;
	
	
    img = flip(videoFrameGray, 2); % Flips the image horizontally
    
    bbox = step(faceDetector, img); % Creating bounding box using faceDetector  
      
    if ~ isempty(bbox)  %if face exists 
        biggest_box=1;     
        for i=1:rank(bbox) %find the biggest face
            if bbox(i,3)>bbox(biggest_box,3)
                biggest_box=i;
            end
        end
        faceImage = imcrop(img,bbox(biggest_box,:)); % extract the face from the image
        bboxeyes = step(eyesDetector, faceImage); % locations of the eyepair using faceDetector
           
        if ~ isempty(bboxeyes)  %check it eyepair is available
            
            biggest_box_eyes=1;     
            for i=1:rank(bboxeyes) %find the biggest eyepair
                if bboxeyes(i,3)>bboxeyes(biggest_box_eyes,3)
                    biggest_box_eyes=i;
                end
            end
             
            bboxeyeshalf=[bboxeyes(biggest_box_eyes,1),bboxeyes(biggest_box_eyes,2),bboxeyes(biggest_box_eyes,3)/3,bboxeyes(biggest_box_eyes,4)];   %resize the eyepair width in half
             
            eyesImage = imcrop(faceImage,bboxeyeshalf(1,:));    %extract the half eyepair from the face image
            eyesImage = imadjust(eyesImage);    %adjust contrast

            r = bboxeyeshalf(1,4)/4;
            [centers, radii, metric] = imfindcircles(eyesImage, [floor(r-r/4) floor(r+r/2)], 'ObjectPolarity','dark', 'Sensitivity', 0.93); % Hough Transform
            [M,I] = sort(radii, 'descend');
                 
            eyesPositions = centers;
            %figure(k);     
            subplot(2,2,2),subimage(eyesImage); hold on;
              
            viscircles(centers, radii,'EdgeColor','b');
                  
            if ~isempty(centers)
                pupil_x=centers(1);
                disL=abs(0-pupil_x);    %distance from left edge to center point
                disR=abs(bboxeyes(1,3)/3-pupil_x);%distance from right edge to center point
                subplot(2,2,4);
                if disL>disR+16
                    subimage(right);
                else if disR>disL
                    subimage(left);
                    else
                       subimage(straight); 
                    end
                end
     
            end          
        end
    end
    set(gca,'XtickLabel',[],'YtickLabel',[]);
	hold off;
end