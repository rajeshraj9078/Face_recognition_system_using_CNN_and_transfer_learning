clc;
clear;

datasetFolder = 'dataset';
if ~exist(datasetFolder, 'dir')
    mkdir(datasetFolder);
end

try
    % Webcam
    cam = webcam(1);
    preview(cam);

    % Counter for limiting the number of frames
    frameCount = 0;
    maxFrames = 200;
    yolov4Detector = yolov4ObjectDetector("tiny-yolov4-coco");

    % Initialize variables for dataset
    imageIndex = 0;
 

    % Track multiple people and save images for the dataset.
    while true
        frame = snapshot(cam);                                   % Capture frame from webcam
        bboxes = helperDetectObjects(yolov4Detector, frame);     % Detect people
        detectedFrame = insertObjectAnnotation(frame, 'rectangle', bboxes, 'Person', 'LineWidth', 3);% Display bounding boxes
        imshow(detectedFrame);
        for i = 1:size(bboxes, 1)                                                                    % Save images with bounding boxes containing people
            personBox = bboxes(i, :);                                                                % Extract each person's bounding box
            personImage = imcrop(frame, personBox);                                                  % Crop the person from the frame
            imageFilename = fullfile(datasetFolder, ['person_' num2str(imageIndex) '.png']);         % Save the cropped image to the dataset folder
            imwrite(personImage, imageFilename);
            imageIndex = imageIndex + 1;                                                             % Increment image index
        end

       
    end
catch ME
    % Error handling
    disp('An error occurred:');
    disp(ME.message);
end

% Release webcam
clear cam;

function box = helperDetectObjects(yolov4Det, frame)
    [box, ~, class] = detect(yolov4Det, frame, 'Threshold', 0.5, 'MinSize', [5 5]);
    box = box(class == "person", :);
end

function key = getkeywait()
    w = waitforbuttonpress;
    key = char(w);
end
