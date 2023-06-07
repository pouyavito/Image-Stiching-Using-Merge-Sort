import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tkinter import filedialog
import os


def get_dataset_imgs():
    folder_selected = filedialog.askdirectory(initialdir="./")
    print(folder_selected)
    dataset_images = os.listdir(folder_selected)
    for img in dataset_images:
        filename, file_extention = os.path.splitext(img)
        while file_extention not in [".jpg", ".png"]:
            print("Selected folder doesn't have JPG or PNG images.")
            quit()
    return folder_selected,dataset_images



def match_images(folder,img_paths):
    path = folder + "/"
    not_visited = img_paths
    visited = []
    for i,img in enumerate(not_visited):
        if i != len(not_visited) - 1:
            print(img,  not_visited[i+1])
            img1 = cv.imread(path + img, cv.IMREAD_GRAYSCALE)
            img2 = cv.imread(path + not_visited[i+1], cv.IMREAD_GRAYSCALE)
            detector = cv.AKAZE_create()
            (kps1, descs1) = detector.detectAndCompute(img1, None)
            (kps2, descs2) = detector.detectAndCompute(img2, None)
            # Match the features
            bf = cv.BFMatcher(cv.NORM_HAMMING)
            matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed
            good = []
            for m,n in matches:
                if m.distance < 1*n.distance:
                    good.append([m])
            print(len(good))
            im3 = cv.drawMatchesKnn(img1, kps1, img2, kps2, good[1:20], None, flags=2)
            plt.imshow(im3),plt.show()
            # cv.imshow(img + not_visited[i+1],im3),cv.waitKey(0)


# def match_images(image1, image2):
#     img1 = cv.imread(image1, cv.IMREAD_GRAYSCALE)
#     img2 = cv.imread(image2, cv.IMREAD_GRAYSCALE)
#     detector = cv.AKAZE_create()
#     (kps1, descs1) = detector.detectAndCompute(img1, None)
#     (kps2, descs2) = detector.detectAndCompute(img2, None)
#     bf = cv.BFMatcher(cv.NORM_HAMMING)
#     matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed
#     good = []
#     for m,n in matches:
#         if m.distance < 1*n.distance:
#             good.append([m])
#     return len(good)

def merge_images(cur_image, next_image, H):
    next_image_matrix = transpose_copy(next_image, "image")
    new_row, new_col = (cur_image.shape[1] + next_image.shape[1], cur_image.shape[0])
    transformed_matrix = np.zeros((new_row, new_col, next_image_matrix.shape[2]))

    # Traverse image pixels to calculate new indices
    for i in range(next_image_matrix.shape[0]):
        for j in range(next_image_matrix.shape[1]):
            dot_product = np.dot(H, [i, j, 1])
            i_match = int(dot_product[0, 0] / dot_product[0, 2] + 0.5)
            j_match = int(dot_product[0, 1] / dot_product[0, 2] + 0.5)
            if 0 <= i_match < new_row and 0 <= j_match < new_col:
                transformed_matrix[i_match, j_match] = next_image_matrix[i, j]

    transformed_next_image = transpose_copy(transformed_matrix, "matrix")
    plt.imshow(transformed_next_image)
    plt.title(''), plt.xticks([]), plt.yticks([])
    plt.show()

    # Find non black pixels in current image and create empty mask
    non_black_mask = np.all(cur_image != [0, 0, 0], axis=-1)
    empty_mask = np.zeros((transformed_next_image.shape[0], transformed_next_image.shape[1]), dtype=bool)
    empty_mask[0:cur_image.shape[0], 0:cur_image.shape[1]] = non_black_mask

    # Assign non black pixels of current image to transformed next image
    transformed_next_image[empty_mask, :] = cur_image[non_black_mask, :]
    transformed_next_image = crop_image(transformed_next_image)
    plt.imshow(cv.cvtColor(transformed_next_image, cv.COLOR_BGR2RGB))
    plt.show()
    return transformed_next_image


def crop_image(image):
    # Crop top if black
    if not np.sum(image[0]):
        return crop_image(image[1:])

    # Crop bottom if black
    elif not np.sum(image[-1]):
        return crop_image(image[:-2])

    # Crop left if black
    elif not np.sum(image[:, 0]):
        return crop_image(image[:, 1:])

    # Crop right if black
    elif not np.sum(image[:, -1]):
        return crop_image(image[:, :-2])

    return image


def transpose_copy(copying, type):
    if type == "matrix":
        return np.transpose(copying.copy(), (1, 0, 2)).astype('uint8')
    else:
        return np.transpose(copying.copy(), (1, 0, 2))


if __name__ == '__main__':
    path, dataset_imgs = get_dataset_imgs()
    match_images(path, dataset_imgs)


# img1 = cv.imread('./HW2_Dataset/pano1/cyl_image00.png', cv.IMREAD_GRAYSCALE)
# img2 = cv.imread('./HW2_Dataset/pano1/cyl_image01.png', cv.IMREAD_GRAYSCALE)
# # gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# # gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)    

# # initialize the AKAZE descriptor, then detect keypoints and extract
# # local invariant descriptors from the image
# detector = cv.AKAZE_create()
# (kps1, descs1) = detector.detectAndCompute(img1, None)
# (kps2, descs2) = detector.detectAndCompute(img2, None)

# print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
# print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))    

# # Match the features
# bf = cv.BFMatcher(cv.NORM_HAMMING)
# matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 1*n.distance:
#         good.append([m])

# # cv2.drawMatchesKnn expects list of lists as matches.
# im3 = cv.drawMatchesKnn(img1, kps1, img2, kps2, good[1:20], None, flags=2)
# cv.imshow("HELLO",im3),cv.waitKey(0)