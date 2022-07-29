
import cv2
import numpy as np
import math

def transform_perspective(img: cv2.Mat, pts1):
    #organize_points(pts1)
    cardH=(math.sqrt((pts1[2][0]-pts1[1][0])**2+(pts1[2][1]-pts1[1][1])**2) + math.sqrt((pts1[0][0]-pts1[3][0])**2+(pts1[0][1]-pts1[3][1])**2))/2
    cardW=(math.sqrt(((pts1[0][0]-pts1[1][0])**2)+(pts1[0][1]-pts1[1][1])**2) + math.sqrt(((pts1[2][0]-pts1[3][0])**2)+(pts1[2][1]-pts1[3][1])**2))/2
    #print(cardH,cardW)
    # cardW=ratio*cardH;
    pts2 = np.float32([[pts1[0][0],pts1[0][1]], [pts1[0][0]+cardW, pts1[0][1]], [pts1[0][0]+cardW, pts1[0][1]+cardH], [pts1[0][0], pts1[0][1]+cardH]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    offsetSize=500
    transformed = np.zeros((int(cardW+offsetSize), int(cardH+offsetSize)), dtype=np.uint8);
    dst = cv2.warpPerspective(img, M, transformed.shape)
    dst=dst[int(pts1[0][1]):int(pts1[0][1]+cardH),int(pts1[0][0]):int(pts1[0][0]+cardW)]
    return dst
def export_to_pdf(cv2_imgs, pdf_name):
    from PIL import Image
    pdf_images=[]
    for img in cv2_imgs:
        #first convert the cv2 images to PIL images
        temp = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(temp)
        pdf_images.append(pil_image)
    #now we save the images to a pdf
    pdf_images[0].save(pdf_name, save_all=True, append_images=pdf_images[1:])

def main():
    img = cv2.imread('test.jpg')
    rows,cols,ch = img.shape
    pts1 = np.float32([[360,50],[2122,470],[2264, 1616],[328,1820]])
    #pts1 = np.float32([[385,473],[2201,81],[2217, 1857],[385,1609]])
    dst=transform_perspective(img,pts1)
    img = cv2.imread('test2.jpg')
    rows,cols,ch = img.shape
    pts1 = np.float32([[433,469],[2205,53],[2264, 1813],[313,1637]])
    dst2=transform_perspective(img,pts1)
    export_to_pdf([dst,dst2],"out.pdf")

if __name__=="__main__":
    main()