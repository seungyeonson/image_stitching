#USAGE
#python image_stitching.py --images images/scottsdale --output output.png --crop 1

from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="stitch할 input 디렉토리 경로")
ap.add_argument("-o", "--output", type=str, required=True, help="output image 만들 경로")
ap.add_argument("-c", "--crop", type=int, default=0, help="보기 좋게 자를지 말지")
args = vars(ap.parse_args())

#이미지를 경로로 받아오고 초기의 작업
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

#이미지를 받아와서 images배열에 저장
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

print("[INFO] stitching images...")
#stitch를 하기 위한 환경설정
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)# stitch를 시도하는 함수

#만약 status가 0이면 성공적으로 image 처리 한 것이다.
if status == 0:
    if args["crop"] > 0:
        print("[INFO] cropping...")
        #기존 사진의 경계의 여백을 늘력준다.
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        #기존 사진의 색깔을 gray로 바꾼다.
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        #반전 시킨다.+grayscale된 이미지를 이분화 한다. ex) 흰검으로
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        #contour란 동일한 색, 동일한 강도를 가지고 있는 영역의 경계선을 말한다.
        #findContours함수는 모든contour line을 찾아 image, contour hierarchy 등 다양한 것을 return 하지만
        #버전마다 return이 다르므로 grabcontour를 해준다.
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #RETR_EXTERNAL:가장 바깥쪽 contour, CHAIN_APPROX_SIMPLE: contour line을 그릴 수 있는 가장 간단한 point
        cnts = imutils.grab_contours(cnts)

        #contour와 관련된 값만을 뽑아내기 위해 사용
        c = max(cnts, key=cv2.contourArea)
        #그냥 검은 mask를 만든다.
        mask = np.zeros(thresh.shape, dtype="uint8")
        #x,y는 첫 시작점, w,h는 길이 너비
        (x, y, w, h) = cv2.boundingRect(c)
        #마스크 위에 사각형 그리는데 흰색으로 그린다.(img, start, end, color, thickness)=>두께 -1이면 채우기
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        #mask의 두개의 copy를 만든다. 하나는 실제 이미지로 사용하고, 다른 하나는 contour로 사용한다.
        minRect = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

        # countNonZero함수는 주어진 이미지의 흰색(0)을 세는 함수=>여기서는 흰색부분이 없어질때까지
        # erode함수는 mask에서 어두운 부분이 점점 커진다.(예시에서는 가운데로 어두워진다)
        # subtract함수는 앞인자에서 뒷영역을 제외한 부분을 뺀다.
        # 결론은 minRect는 뒷배경의 검은부분이 점점늘어나고 sub는 minRect에서 thresh를 제외한 것을 뺀것인데 흰부분이 점점사라진다.
        # 이 함수의 역할은 stitching 함수를 result할때 나오는 이미지를 보기 좋은 이미지로 출력하기 위함이다.
        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)

        # use the bounding box coordinates to extract the our final
        # stitched image
        stitched = stitched[y:y + h, x:x + w]

        cv2.imwrite(args["output"], stitched)


        cv2.imshow("stitched", stitched)
        cv2.waitKey(0)
else:
	print("[INFO] image stitching failed ({})".format(status))

