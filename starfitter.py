# -*- coding: utf-8 -*-

import cv2
from glob import glob
from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos
import numpy as np
from scipy.optimize import minimize
from json import dump
import os

# +y is north, +x, is east
# quadratic pixels are assumed
# a radial lens distortion model is used, with a n-th order polynomial

camid = 1
in_size = (1080//2, 1920//2)
roll = 0
year = 2022
month = 2
day = 12
image_dates = ([(day, h) for h in range(18, 24)] +
               [(day + 1, h) for h in range(6)])
image_input_dir = "fit_images"
output_data_dir = "fitres_AMS131"
latlon = (69.662, 18.936)
fov_x = 0.8
fov = (fov_x, fov_x*in_size[0]/in_size[1])
poly_order = 3
initial_guess = [
    {"id": 1, "fov": fov, "c_2": 0, "c_3": 0,
     "az": np.pi*90/180, "alt": np.pi*15/180, "roll": 0},
    {"id": 2, "fov": fov, "c_2": 0, "c_3": 0,
     "az": np.pi*150/180, "alt": np.pi*17/180, "roll": 0}
]
fitstars = [
    (1, (2022, 2, 12, 20, 10, 1), 67927, (259, 309)),
    (1, (2022, 2, 12, 20, 10, 1), 49669, (900, 106)),
    (1, (2022, 2, 12, 23, 55, 0), 69673, (812, 62)),
    (1, (2022, 2, 13, 1, 45, 0), 91262, (371, 32)),
   
    (2, (2022, 2, 12, 23, 30, 1), 69673, (103, 91)),
    (2, (2022, 2, 12, 23, 30, 1), 69673, (103, 91)),
    (2, (2022, 2, 12, 23, 30, 1), 57632, (501, 89)),
    (2, (2022, 2, 12, 23, 30, 1), 49669, (782, 78)),
    (2, (2022, 2, 12, 20, 10, 0), 50583, (190, 72)),
    (2, (2022, 2, 12, 20, 10, 0), 37279, (658, 175)),    
    (2, (2022, 2, 13, 1, 25, 0), 69673, (406, 49)),
    (2, (2022, 2, 13, 1, 25, 0), 67927, (465, 50)),
    (2, (2022, 2, 13, 1, 25, 0), 63608, (627, 113)),
    (2, (2022, 2, 13, 1, 25, 0), 61941, (708, 248)),
    (2, (2022, 2, 13, 1, 25, 0), 65474, (598, 378)),
]
second_pass_files = [
    "2022_02_12_20_10_01_1.jpg",
    "2022_02_12_20_40_01_1.jpg",
    "2022_02_12_21_05_01_1.jpg",
    "2022_02_12_21_10_00_1.jpg",
    "2022_02_12_22_55_00_1.jpg",
    "2022_02_12_23_00_00_1.jpg",
    "2022_02_12_23_25_01_1.jpg",
    "2022_02_12_23_40_00_1.jpg",
    "2022_02_12_23_55_00_1.jpg",
    "2022_02_13_01_40_00_1.jpg",
    "2022_02_13_01_50_00_1.jpg",
    "2022_02_13_02_40_01_1.jpg",
    
    "2022_02_12_20_30_01_2.jpg",
    "2022_02_12_20_40_01_2.jpg",
    "2022_02_12_20_50_00_2.jpg",
    "2022_02_12_22_00_01_2.jpg",
    "2022_02_12_23_00_00_2.jpg",
    "2022_02_12_23_10_00_2.jpg",
    "2022_02_12_23_50_00_2.jpg"
]
detector_settings = {
    "min_area": 4,
    "match_dist": 20,
    1: {"region_x": (0, 1), "region_y": (0, 0.6), "filter_size": 7, "threshold": 10},
    2: {"region_x": (0, 1), "region_y": (0, 0.6), "filter_size": 7, "threshold": 15}
}
cams_to_fit = [1]
cam_to_show = 1
f = load.open(hipparcos.URL)
df = hipparcos.load_dataframe(f)
df = df[df['magnitude'] <= 3]
observer = wgs84.latlon(*latlon)
planets = load('de421.bsp')
earth = planets['earth']


def get_matrix(cam):
    return np.matmul(Rz(cam["az"]),
                     np.matmul(Rx(np.pi/2 - cam["alt"]), Rz(cam["roll"])))


def Rz(phi):
    return np.array([[ np.cos(phi), np.sin(phi), 0],
                     [-np.sin(phi), np.cos(phi), 0],
                     [           0,           0, 1]])


def Rx(phi):
    return np.array([[1,            0,           0],
                     [0,  np.cos(phi), np.sin(phi)],
                     [0, -np.sin(phi), np.cos(phi)]])


def pixel_to_vec(cam, px, py): 
    fov = cam["fov"]
    x = 2*px/in_size[1] - 1
    y = 2*py/in_size[0] - 1
    y = y*fov[1]/fov[0]
    if poly_order >= 2:
       r = np.sqrt(x**2 + y**2)
       r_corr = r + sum(cam[f"c_{n}"]*r**n for n in range(2, poly_order + 1))       
       x, y = x*r_corr/r, y*r_corr/r
    v = np.matmul(get_matrix(cam), np.array([x*fov[0], y*fov[0], 1]))
    return v/np.linalg.norm(v)


def vec_to_pixel(cam, v):
    v = np.matmul(get_matrix(cam).T, v)
    fov = cam["fov"]
    x, y = v[0]/(fov[0]*v[2]), v[1]/(fov[0]*v[2])
    if poly_order >= 2:
        r_corr = np.sqrt(x**2 + y**2)
        roots = np.roots([cam[f"c_{poly_order - i}"]
                          for i in range(poly_order - 1)] + [1, -r_corr])
        r = next(np.real(root) for root in roots
                 if (np.abs(np.imag(root)) <= np.finfo(float).eps and
                     np.real(root) > 0))
        x, y = x*r/r_corr, y*r/r_corr
    y = y*fov[0]/fov[1]
    px = in_size[1]*(x + 1)/2
    py = in_size[0]*(y + 1)/2
    return round(px), round(py)


def calculate_star_pos(star_data, t):
    star = Star.from_dataframe(star_data)
    apparent = (earth + observer).at(t).observe(star).apparent()
    alt, az, distance = apparent.altaz()
    az, alt = az.radians, alt.radians
    x = np.sin(az)*np.cos(alt)
    y = np.cos(az)*np.cos(alt)
    z = np.sin(alt)
    return [x, y, z]


def plot_stars(cam, img, datetime):
    t = load.timescale().utc(*datetime)
    for i, star_data in df.iterrows():
        px, py = vec_to_pixel(cam, calculate_star_pos(star_data, t))
        if px >= 0 and py >= 0 and px < in_size[1] and py < in_size[0]:
            cv2.circle(img, (px, py), 5, (255, 255, 255))
            cv2.putText(img, f"{i}", (px + 7, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def list_fitstars(cam, datetime):
    return [fs for fs in fitstars if fs[0] == cam["id"] and fs[1] == datetime]


def plot_residuals(cam, img, datetime):
    t = load.timescale().utc(*datetime)
    plotted = False
    for fitstar in list_fitstars(cam, datetime):
        star = vec_to_pixel(cam, calculate_star_pos(df.loc[fitstar[2]], t))
        cv2.circle(img, fitstar[3], 5, (0, 255, 0))
        cv2.line(img, star, fitstar[3], (255,255, 0), 1)
        plotted = True
    return plotted
        

def detect_stars(cam, org_img, img):
    settings = detector_settings[cam["id"]]
    rx = settings["region_x"]
    ry = settings["region_y"]
    hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)
    hsv = hsv[round(ry[0]*org_img.shape[0]):round(ry[1]*org_img.shape[0]),
              round(rx[0]*org_img.shape[1]):round(rx[1]*org_img.shape[1]), 2]
    hsv = cv2.subtract(hsv, cv2.medianBlur(hsv, settings["filter_size"]))
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = settings["threshold"]
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = detector_settings["min_area"]
    params.filterByCircularity = False
    params.minCircularity = 0
    params.filterByConvexity = False
    params.minConvexity = 0
    params.filterByColor = True
    params.blobColor = 255
    params.filterByInertia = False
    params.minInertiaRatio = 0
    stars = cv2.SimpleBlobDetector_create(params).detect(hsv)
    res = []
    for star in stars:
        x = star.pt[0] - round(rx[0]*org_img.shape[1])
        y = star.pt[1] - round(ry[0]*org_img.shape[0])
        x = round(x/org_img.shape[1]*in_size[1])
        y = round(y/org_img.shape[0]*in_size[0])
        res.append((x, y))
        if img is not None:
            cv2.circle(img, (x, y), 5, (0, 0, 255))
    return res


def match_stars(cam, img, datetime, detected_stars):
    match_dist = detector_settings["match_dist"]
    t = load.timescale().utc(*datetime)
    stars = []
    for i, star_data in df.iterrows():
        px, py = vec_to_pixel(cam, calculate_star_pos(star_data, t))
        if px < 0 or py < 0 or px >= in_size[1] or py >= in_size[0]:
            continue
        stars.append((px, py, star_data.name))
    result = []
    for pdx, pdy in detected_stars:
        mindist = match_dist**2
        for i, (px, py, name) in enumerate(stars):
            dist = (px - pdx)**2 + (py - pdy)**2
            if dist < mindist:
                mindist = dist
                closest_star = i
        if mindist < match_dist**2:
            px, py, name = stars[closest_star]
            del stars[closest_star]
            result.append((camid, datetime, name, (pdx, pdy)))
            if img is not None:
                cv2.line(img, (pdx, pdy), (px, py), (0, 255, 255), 1)
    return result


def info_from_file(file):
    file = file.split(".")[0]
    info = tuple([int(t) for t in file.split("/")[-1].split("_")[:7]])
    return info[:6], info[6]


def show_file(cam, file, residuals_only):
    datetime, camid = info_from_file(file)
    if residuals_only and not list_fitstars(cam, datetime):
        return False
    org_img = cv2.imread(file)
    img = cv2.resize(org_img, (in_size[1], in_size[0]))
    plot_residuals(cam, img, datetime)
    plot_stars(cam, img, datetime)
    cv2.putText(img, file, (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    detected_stars = detect_stars(cam, org_img, img)
    match_stars(cam, img, datetime, detected_stars)
    cv2.imshow("preview", img)
    return True


def show(cam, residuals_only=True):
    cv2.namedWindow("preview")
    cv2.moveWindow("preview", 50, 50)
    for day, hour in image_dates:
        file = f"{year:04d}_{month:02d}_{day:02d}_{hour:02d}_*_{cam['id']}.jpg"
        files = glob(os.path.join(image_input_dir, file))
        files.sort()
        for file in files:
            if not show_file(cam, file, residuals_only):
                continue
            while True:
                key = cv2.waitKey(0)
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                if key == ord("n"):
                    break
    cv2.destroyAllWindows()


def cam_by_camid(calib_data, camid):
    return next(cam for cam in calib_data if cam["id"] == camid)


def params_to_calib_data(params):
    fov = (params[0], params[0]*in_size[0]/in_size[1])
    pcnt = max([0, poly_order - 2])
    res = []
    for i, camid in enumerate(cams_to_fit):
        cam = {"id": camid, "fov": fov, "az": params[pcnt*i + 1],
               "alt": params[pcnt*i + 2], "roll": params[pcnt*i + 3]}
        for o in range(2, poly_order + 1):
            cam[f"c_{o}"] = params[pcnt*i + 2 + o]
        res.append(cam)
    return res


def calib_data_to_params(calib_data):
    params = []
    for i, camid in enumerate(cams_to_fit):
        cam = cam_by_camid(calib_data, camid)
        if i == 0:
            params.append(cam["fov"][0])
        params.append(cam["az"])
        params.append(cam["alt"])
        params.append(cam["roll"])
        for i in range(2, poly_order + 1):
            params.append(cam[f"c_{i}"])
    return params


def calc_chi2(params, fitstars):
    calib_data = params_to_calib_data(params)
    chi2 = 0
    for camid, time, star, (px, py) in fitstars:
        if camid not in cams_to_fit:
            continue
        cam = cam_by_camid(calib_data, camid)
        ppos = pixel_to_vec(cam, px, py)
        spos = calculate_star_pos(df.loc[star], load.timescale().utc(*time))
        chi2 += np.linalg.norm(ppos - spos)**2
    return chi2


def do_fit(fitstars):
    fitres = minimize(calc_chi2, calib_data_to_params(initial_guess),
                      (fitstars))
    calib_data_result = params_to_calib_data(fitres.x)
    fname = "fit_" + "_".join([f"{camid}" for camid in cams_to_fit]) + ".json"
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    with open(os.path.join(output_data_dir, fname), 'w') as outfile:
        dump(calib_data_result, outfile, indent=6)
    return calib_data_result


def extract_additional_fitstars(fitstars):
    additional_fitstars = set(fitstars)
    for file in second_pass_files:
        datetime, camid = info_from_file(file)
        if camid not in cams_to_fit:
            continue
        cam = cam_by_camid(calib_data_result, camid)
        org_img = cv2.imread(os.path.join(image_input_dir, file))
        detected_stars = detect_stars(cam, org_img, None)
        res = match_stars(cam, None, datetime, detected_stars)
        additional_fitstars.update(res)
    return list(additional_fitstars)
    

#%%
show(cam_by_camid(initial_guess, cam_to_show), True)
#%%
calib_data_result = do_fit(fitstars)
#%%
show(cam_by_camid(calib_data_result, cam_to_show), False)
#%%
additional_fitstars = extract_additional_fitstars(fitstars)
#%%
calib_data_final = do_fit(additional_fitstars)
#%%
show(cam_by_camid(calib_data_final, cam_to_show), False)

