import os
import joblib
import numpy as np
from glob import glob
from tensorflow.keras.models import load_model
from ncwrite import nc_write
import matplotlib
import seaborn as sns
import xarray as xr

matplotlib.use('agg')
sns.set_style(style='ticks')

def img_est_chl(data, model, x_scaler, y_scaler):
    """Takes any number of input bands (shaped [Height, Width]) and
    returns the products for that image, in the same shape."""
    expected_features = 11
    assert (data.shape[-1] == expected_features), (
        f'Got {data.shape[-1]} features; expected {expected_features} features for VIIRS sensor')
    im_shape = data.shape[:-1]
    im_data = data.reshape((-1, expected_features))
    # im_data = data.reshape((-1, expected_features))
    x_test = x_scaler.transform(im_data)
    # x_test[x_test<=0] = np.nan
    # x_test = np.nan_to_num(x_test)
    # y_hat = model.predict(x_test).reshape(-1, 1)
    y_hat = model(x_test, training=False)
    est = np.exp(y_scaler.inverse_transform(y_hat))
    chl = est.reshape(1, im_shape[0], im_shape[1])
    chl[chl < 0.01] = np.nan
    chl[chl > 1000] = np.nan
    model = None
    im_data = None
    del model
    return chl

if __name__ == '__main__':
    # load model and scalers
    waves = [412, 443, 490, 510, 560, 620, 665, 681, 709, 754, 779]
    x_scaler = joblib.load(
        'benchmark/x_scaler_final.json')
    y_scaler = joblib.load(
        'benchmark/y_scaler_final.json')
    model_path = 'benchmark/OLCI-MERIS_Chla_DNN_final.h5'
    #
    working_path = r'Z:\level1_data\Sentinel3\Global_Examples\EastAfrica'
    out_dir = r'X:\Chla\Global_Chla_results\Chla'
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    nc_files = glob(working_path + os.path.sep + '*EastAfricanLakes.nc')
    nc_files.sort()
    for n, nc_file in enumerate(nc_files):
        base_file = os.path.basename(nc_file)
        out_file = os.path.join(out_dir, base_file.replace('.nc', '_Chla_DNN.nc'))
        if os.path.exists(out_file):
            print(out_file + ' existing. skip...')
            continue
        # read netcdf4 data
        # try:
        ds = xr.open_dataset(nc_file)
        # print(ds)
        data = None
        for w, wave in enumerate(waves):
            Rw = ds[f'Rw{wave}'].values
            if w == 0:
                data = np.zeros((Rw.shape[0], Rw.shape[1], len(waves)))
            # calculte Rw to Rrs
            Rrs = Rw / np.pi
            data[:, :, w] = Rrs
        dnn_model = load_model(model_path)
        chl_dnn = img_est_chl(data, dnn_model, x_scaler, y_scaler)
        #
        nc_write(out_file, 'lat', data=ds['latitude'].values, new=True)
        nc_write(out_file, 'lon', data=ds['longitude'].values, new=False)
        nc_write(out_file, 'chla_dnn', data=chl_dnn, new=False)
        #
        ds = None
        data = None
        dnn_model = None
        chl_dnn = None
        #
        n = n + 1
        print(out_file)
        # print('>{}: {}/{}: {} has been written at {}'.format(lake_id, n+1, len(nc_files),
        #     out_file, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))