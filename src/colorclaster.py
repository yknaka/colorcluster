# coding: UTF-8
import sys
import consoleoptions as get_option
import pandas as pd
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


def pil2cv(image):
  new_image = np.array(image, dtype=np.uint8)
  if new_image.ndim == 2:  # monochrome
    pass
  elif new_image.shape[2] == 3:  # color
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
  elif new_image.shape[2] == 4:  # transparency
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
  return new_image


def main(args=sys.argv):
  # initial value of the options
  optionDict = {}
  optionDict["input_fig_path"] = None
  optionDict["color_num"] = 8
  optionDict["export_fig_path"] = "result.jpg"
  optionDict["export_array_path"] = "array.csv"

  if type(args) is str:
    args = [args]
  optionDict = get_option.get_dict(args, optionDict)
  get_option.to_int(optionDict, "color_num")
  print("loading image")
  if optionDict["input_fig_path"] is None:
    print("No Image.")
    sys.exit(0)
  cv2_img = cv2.imread(optionDict["input_fig_path"])
  if cv2_img is None:
    print("Failed to load image!")
    sys.exit(0)
  cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
  img = cv2_img.reshape((cv2_img.shape[0] * cv2_img.shape[1], 3))
  print("KMeans color clustering...")
  cluster = KMeans(n_clusters=optionDict["color_num"]).fit(img)
  cluster_centers = cluster.cluster_centers_.astype(np.uint8)
  print("repainting image")
  repaint_img = np.zeros(img.shape).astype(np.uint8)
  for i, color_arr in enumerate(repaint_img):
    color_arr[:] = cluster_centers[cluster.labels_[i]][:]
  repaint_img2 = repaint_img.reshape((cv2_img.shape[0], cv2_img.shape[1], 3))
  img_cl = Image.fromarray(repaint_img2)
  img_cl.save(optionDict["export_fig_path"])
  if "export_color_array" in optionDict:
    df_cluster = pd.DataFrame(cluster_centers, columns=["R", "G", "B"])
    df_cluster.to_csv(optionDict["export_array_path"])
  if "show_img" in optionDict:
    while True:
      cv2.imshow('color cluster', pil2cv(img_cl))
      k = cv2.waitKey(1)
      # ESC key to close
      if k == 27:
        break
    cv2.destroyWindow('color cluster')
    cv2.waitKey(1)


if __name__ == '__main__':
  main()
