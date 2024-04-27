import PIL.ImageShow
import cv2
import numpy as np
import matplotlib.pyplot as plt


def transform_perspective(img: cv2.Mat, pts1):
    width = euclidean_distance(
        np.float32([euclidean_distance(pts1[0] - pts1[1]), euclidean_distance(pts1[2] - pts1[3])]))
    height = euclidean_distance(
        np.float32([euclidean_distance(pts1[0] - pts1[2]), euclidean_distance(pts1[1] - pts1[3])]))

    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, matrix, [int(width),int(height)])
    plt.imshow(img)
    plt.show()
    plt.imshow(dst)
    plt.show()
    return dst


# returns the Euclidean distance of a vector, that is sqrt(a**2+b**2+...)
#   REQUIRES vector to numpy array, and has only number(int, float, etc.) children
def euclidean_distance(v1: np.ndarray):
    return np.sqrt(sum(v1 ** 2))


def export_to_pdf(cv2_imgs, pdf_name):
    from PIL import Image
    pdf_images = []
    for img in cv2_imgs:
        # first convert the cv2 images to PIL images
        temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(temp)

        pdf_images.append(pil_image)

    # now we save the images to a pdf
    pdf_images[0].save(pdf_name, save_all=True, append_images=pdf_images[1:])


def main():
    img = cv2.imread('test.jpg')
    pts1 = np.float32([[360, 50], [2122, 470], [328, 1820], [2264, 1616]])
    # pts1 = np.float32([[385,473],[2201,81],[2217, 1857],[385,1609]])
    dst = transform_perspective(img, pts1)

    img = cv2.imread('test2.jpg')
    pts1 = np.float32([[433, 469], [2205, 53], [313, 1637], [2264, 1813]])
    dst2 = transform_perspective(img, pts1)
    export_to_pdf([dst, dst2], "out.pdf")


if __name__ == "__main__":
    main()
