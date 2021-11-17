import numpy as np


def haversine_vectorized(df,
                         start_lat="pickup_latitude",
                         start_lon="pickup_longitude",
                         end_lat="dropoff_latitude",
                         end_lon="dropoff_longitude"):
    """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).
        Vectorized version of the haversine distance for pandas df
        Computes distance in kms
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)),\
        np.radians(df[start_lon].astype(float))
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)),\
        np.radians(df[end_lon].astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) *\
        np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def minkowski_distance(x1, x2, y1, y2, p):
    delta_x = x1 - x2
    delta_y = y1 - y2
    return ((abs(delta_x)**p) + (abs(delta_y))**p)**(1 / p)


def deg2rad(coordinate):
    return coordinate * np.pi / 180


# convert radians into distance
def rad2dist(coordinate):
    earth_radius = 6371  # km
    return earth_radius * coordinate


# correct the longitude distance regarding the latitude (https://jonisalonen.com/2014/computing-distance-between-coordinates-can-be-simple-and-fast/)
def lng_dist_corrected(lng_dist, lat):
    return lng_dist * np.cos(lat)


def minkowski_distance_gps(df,
                           p,
                           start_lat="pickup_latitude",
                           start_lon="pickup_longitude",
                           end_lat="dropoff_latitude",
                           end_lon="dropoff_longitude"):
    lat1, lat2, lon1, lon2 = [
        deg2rad(coordinate) for coordinate in
        [df[start_lat], df[start_lon], df[end_lat], df[end_lon]]
    ]
    y1, y2, x1, x2 = [rad2dist(angle) for angle in [lat1, lat2, lon1, lon2]]
    x1, x2 = [
        lng_dist_corrected(elt['x'], elt['lat']) for elt in [{
            'x': x1,
            'lat': lat1
        }, {
            'x': x2,
            'lat': lat2
        }]
    ]
    return minkowski_distance(x1, x2, y1, y2, p)


def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())
