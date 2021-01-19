# import cv2
# import numpy as np

# def resize_norm_img_srn(img, image_shape):
#     imgC, imgH, imgW = image_shape
#     img_black = np.zeros((imgH, imgW))
#     im_hei = img.shape[0]
#     im_wid = img.shape[1]

#     if im_wid <= im_hei * 1:
#         img_new = cv2.resize(img, (imgH * 1, imgH))
#     elif im_wid <= im_hei * 2:
#         img_new = cv2.resize(img, (imgH * 2, imgH))
#     elif im_wid <= im_hei * 3:
#         img_new = cv2.resize(img, (imgH * 3, imgH))
#     else:
#         img_new = cv2.resize(img, (imgW, imgH))

#     img_np = np.asarray(img_new)
#     img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#     img_black[:, 0:img_np.shape[1]] = img_np
#     img_black = img_black[:, :, np.newaxis]

#     row, col, c = img_black.shape
#     print("d")
#     print("row: {}, col: {}".format(row, col))
#     c = 1
#     cv2.imwrite("2.jpg", img_black)
#     return np.reshape(img_black, (c, row, col)).astype(np.float32)

# img = cv2.imread("1.jpg")
# resize_norm_img_srn(img, [3,48,192])

# f_wrong = open('wrong.txt', 'w')
# f_wrong.writelines("dadadada")

save_to_root = None
if save_to_root == None:
	print("x")