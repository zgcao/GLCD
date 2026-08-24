# Data and model provenance

## Demo data

`demo_data/erie_olci_demo.nc` is a 64 x 64 pixel subset of a Sentinel-3A OLCI
scene over Lake Erie acquired on 4 July 2023. The source product identifier is:

`S3A_OL_1_EFR____20230704T152718_20230704T153018_20230705T161721_0179_100_339_2160_PS1_O_NT_003.SEN3`

The included variables are only latitude, longitude, and the 11 water
reflectance bands required by the model. Water reflectance was generated from
the Sentinel-3 Level-1 product with POLYMER 4.16. The subset was selected solely
to make the reviewer demo small and fast. Copernicus/Sentinel source data remain
subject to their applicable data-access and attribution terms; the MIT license
in this repository applies to the software, not to third-party source data.

## Model assets

`model/OLCI-MERIS_Chla_DNN_final.h5` contains the trained Keras DNN used for
inference. The two `.joblib` files contain the fitted input RobustScaler and
output MinMaxScaler. They are required to reproduce the numerical inference.
The author-generated model and scaler assets are distributed with this package
under the repository MIT License. The license does not supersede third-party
terms applicable to the Sentinel/Copernicus-derived demo data.
