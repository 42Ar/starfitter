# -*- coding: utf-8 -*-

import cv2
from glob import glob
from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos
import numpy as np
from scipy.optimize import minimize
import json
import os

# +y is north, +x, is east
# quadratic pixels are assumed
# a radial lens distortion model is used, with a n-th order polynomial

camid = 1
in_size = (1080//2, 1920//2)
roll = 0
year = 2022
month = 2
day = 13
image_dates = ([(day, h) for h in range(18, 24)] +
               [(day + 1, h) for h in range(6)])
image_input_dir = "fit_images"
output_data_dir = "fitres_AMS131"
latlon = (69.662, 18.936)
c_2 = -0.4
c_3 =  0.62
fov_x = 0.79
fov = (fov_x, fov_x*in_size[0]/in_size[1])
poly_order = 3
initial_guess = [
    {"id": 1, "fov": fov, "c_2": c_2, "c_3": c_3,
     "az": np.pi*75/180, "alt": np.pi*15/180, "roll": 0},
    {"id": 2, "fov": fov, "c_2": c_2, "c_3": c_3,
     "az": np.pi*150/180, "alt": np.pi*17/180, "roll": 0},
    {"id": 3, "fov": fov, "c_2": c_2, "c_3": c_3,
     "az": np.pi*220/180, "alt": np.pi*17/180, "roll": 0}
]
fitstars = [
    (1, (2022, 2, 12, 21, 15, 0), 72105, (323, 193)),
    (1, (2022, 2, 12, 21, 15, 0), 69673, (426, 254)),
    (1, (2022, 2, 12, 21, 15, 0), 54872, (844, 21)),

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

    (3, (2022, 2, 13, 18, 40, 1), 9884, (612, 82)),
    (3, (2022, 2, 13, 18, 40, 1), 8903, (637, 121))
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
    "2022_02_12_23_50_00_2.jpg",

    "2022_02_13_18_40_01_3.jpg",
    "2022_02_13_18_45_01_3.jpg",
    "2022_02_13_18_50_01_3.jpg",
    "2022_02_13_19_00_00_3.jpg",
    "2022_02_13_19_05_01_3.jpg",
    "2022_02_13_19_10_01_3.jpg",
    "2022_02_13_19_15_02_3.jpg",
    "2022_02_13_19_30_00_3.jpg",
    "2022_02_13_19_35_00_3.jpg",
    "2022_02_13_19_40_00_3.jpg",
    "2022_02_13_19_45_00_3.jpg",
    "2022_02_13_19_50_01_3.jpg",
    "2022_02_13_21_35_00_3.jpg",
    "2022_02_13_22_10_01_3.jpg",
    "2022_02_13_22_15_01_3.jpg",
    "2022_02_13_22_40_01_3.jpg"
]
detector_settings = {
    "min_area": 4,
    "match_dist": 20,
    1: {"region_x": (0, 1), "region_y": (0, 0.5), "filter_size": 7, "threshold": 10},
    2: {"region_x": (0, 1), "region_y": (0, 0.6), "filter_size": 7, "threshold": 15},
    3: {"region_x": (0, 1), "region_y": (0, 0.75), "filter_size": 7, "threshold": 10, "ignore": [(220, 175, 324, 533)]},
    4: {"region_x": (0, 1), "region_y": (0, 0.6), "filter_size": 7, "threshold": 10}
}
cams_to_fit = [4]
cam_to_show = 4
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
    x = 2*px/(in_size[1] - 1) - 1
    y = 2*py/(in_size[0] - 1) - 1
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
    px = (in_size[1] - 1)*(x + 1)/2
    py = (in_size[0] - 1)*(y + 1)/2
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
    hsv = hsv[round(ry[0]*(org_img.shape[0] - 1)):1 + round(ry[1]*(org_img.shape[0] - 1)),
              round(rx[0]*(org_img.shape[1] - 1)):1 + round(rx[1]*(org_img.shape[1] - 1)), 2]
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
        x = star.pt[0] - round(rx[0]*(org_img.shape[1] - 1))
        y = star.pt[1] - round(ry[0]*(org_img.shape[0] - 1))
        x = round(x/(org_img.shape[1] - 1)*(in_size[1] - 1))
        y = round(y/(org_img.shape[0] - 1)*(in_size[0] - 1))
        if any(x > ignore_rect[0] and y > ignore_rect[1] and
               x < ignore_rect[2] and y < ignore_rect[3]
               for ignore_rect in settings["ignore"]):
            continue
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
            result.append((cam["id"], datetime, name, (pdx, pdy)))
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
        file = f"{year:04d}_{month:02d}_{day:02d}_{hour:02d}_*_{cam['id']}."
        files = glob(os.path.join(image_input_dir, file + "jpg"))
        files += glob(os.path.join(image_input_dir, file + "png"))
        files.sort()
        for file in files:
            if not show_file(cam, file, residuals_only):
                continue
            while True:
                key = cv2.waitKey(0)
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                if key == ord("f"):
                    print(f'"{file.split["/"][-1]}",')
                    break
                if key == ord("n"):
                    break
    cv2.destroyAllWindows()


def cam_by_camid(calib_data, camid):
    return next(cam for cam in calib_data if cam["id"] == camid)


def params_to_calib_data(params, orig_calib_data, fix_lens_params):
    p_shared = 0
    if not fix_lens_params:
        fov = (params[0], params[0]*in_size[0]/in_size[1])
        p_shared = 1 + max([poly_order - 1, 0])
    res = []
    for i, camid in enumerate(cams_to_fit):
        new_calib = dict(cam_by_camid(orig_calib_data, camid))
        new_calib["az"] = params[3*i + p_shared]
        new_calib["alt"] = params[3*i + p_shared + 1]
        new_calib["roll"] = params[3*i + p_shared + 2]
        if not fix_lens_params:
            new_calib["fov"] = fov
            for o in range(2, poly_order + 1):
                new_calib[f"c_{o}"] = params[o - 1]
        res.append(new_calib)
    return res


def calib_data_to_params(calib_data, fix_lens_params):
    params = []
    if not fix_lens_params:
        cam = cam_by_camid(calib_data, cams_to_fit[0])
        params.append(cam["fov"][0])
        for i in range(2, poly_order + 1):
            params.append(cam[f"c_{i}"])
    for camid in cams_to_fit:
        cam = cam_by_camid(calib_data, camid)
        params.append(cam["az"])
        params.append(cam["alt"])
        params.append(cam["roll"])
    return params


def calc_chi2(params, fitstars, fix_lens_params, orig_calib_data):
    calib_data = params_to_calib_data(params, orig_calib_data, fix_lens_params)
    chi2 = 0
    for camid, time, star, (px, py) in fitstars:
        if camid not in cams_to_fit:
            continue
        cam = cam_by_camid(calib_data, camid)
        ppos = pixel_to_vec(cam, px, py)
        spos = calculate_star_pos(df.loc[star], load.timescale().utc(*time))
        chi2 += np.linalg.norm(ppos - spos)**2
    return chi2


def do_fit(fitstars, initial_guess, fix_lens_params):
    p0 = calib_data_to_params(initial_guess, fix_lens_params)
    fitres = minimize(calc_chi2, p0, (fitstars, fix_lens_params,
                                      initial_guess))
    calib_data_result = params_to_calib_data(fitres.x, initial_guess,
                                             fix_lens_params)
    fname = "fit_" + "_".join([f"{camid}" for camid in cams_to_fit]) + ".json"
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    with open(os.path.join(output_data_dir, fname), 'w') as outfile:
        json.dump(calib_data_result, outfile, indent=6)
    return calib_data_result


def extract_additional_fitstars(calib_data, fitstars):
    additional_fitstars = set(fitstars)
    for file in second_pass_files:
        datetime, camid = info_from_file(file)
        if camid not in cams_to_fit:
            continue
        cam = cam_by_camid(calib_data, camid)
        org_img = cv2.imread(os.path.join(image_input_dir, file))
        detected_stars = detect_stars(cam, org_img, None)
        res = match_stars(cam, None, datetime, detected_stars)
        additional_fitstars.update(res)
    return list(additional_fitstars)


def analyze_residuals_stacked(fitstars, calib_data, residual_scale=10):
    plotimg = np.zeros((*in_size, 3))
    for camid, time, star, (px, py) in fitstars:
        if camid not in cams_to_fit:
            continue
        cam = cam_by_camid(calib_data, camid)
        spos = calculate_star_pos(df.loc[star], load.timescale().utc(*time))
        spos = vec_to_pixel(cam, spos)
        cv2.arrowedLine(plotimg, (px, py),
                        (px + (spos[0] - px)*residual_scale,
                         py + (spos[1] - py)*residual_scale), (255, 0, 0), 1)
        cv2.circle(plotimg, spos, 5, (0, 0, 255))
    cv2.imshow("stacked residuals", plotimg)
    cv2.waitKey()
    cv2.destroyAllWindows()


def load_fit_results():
    fname = "fit_" + "_".join([f"{camid}" for camid in cams_to_fit]) + ".json"
    with open(os.path.join(output_data_dir, fname), 'r') as infile:
        return json.loads(infile.read())

#%%
show(cam_by_camid(initial_guess, cam_to_show), residuals_only=True)
#%%
calib_data_result = do_fit(fitstars, initial_guess, fix_lens_params=True)
#%%
show(cam_by_camid(calib_data_result, cam_to_show), residuals_only=False)
#%%
additional_fitstars = extract_additional_fitstars(calib_data_result, fitstars)
#%%
calib_data_final = do_fit(additional_fitstars, calib_data_result, fix_lens_params=False)
#%%
show(cam_by_camid(calib_data_final, cam_to_show), residuals_only=True)
#%%
analyze_residuals_stacked(additional_fitstars, calib_data_final)
